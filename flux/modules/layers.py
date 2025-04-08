import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from flux.math import attention, rope,apply_rope,replace_cross_attn_ci_regions

import os

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

class ReshapeHeadsToSeq(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保输入是四维张量
        assert len(x.shape) == 4, f"Input tensor must be 4-dimensional, got {len(x.shape)} dimensions"
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # 交换 "num_heads" 和 "seq_len" 维度
        x = x.transpose(1, 2)  # 输出形状: (batch_size, seq_len, num_heads, head_dim)
        
        # 合并最后两个维度 (num_heads * head_dim)
        x = x.reshape(batch_size, seq_len, num_heads * head_dim)
        
        return x
    
class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, info) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)

        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2) #[8, 24, 512, 128] + [8, 24, 900, 128] -> [8, 24, 1412, 128]
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        # import pdb;pdb.set_trace()
        attn = attention(q, k, v, pe=pe)
 
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)
        self.reshape = ReshapeHeadsToSeq()

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, info) -> tuple[Tensor, dict]:
        '''这里我给改了,如果报错,改为-> Tensor'''
        mod, _ = self.modulation(vec) # vec[1,3072]
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift # x_mod[1,1772,3072]
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # Save the features in the memory
        
       
        '''此处是我们加的代码'''
        '''此处qkv还没有加上位置信息RoPE'''
        if pe is not None:
           q,k = apply_rope(q,k,pe)
           '''此处apply_rope是Fireflow的代码,我没有改'''
        
        scale_factor = 1 / math.sqrt(q.shape[-1])
        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1) #1*24*1772*1772
        if info['type_s'] == 'edit':
            if info['inject'] and info['id'] <= info['end_layer_index'] and info['id'] >= info['start_layer_index']:
                ci_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'ci' + '_' + info['k']
                ic_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'ic' + '_' + info['k']
                ii_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'ii' + '_' + info['k']
                cc_feature_name = str(info['t']) + '_' + str(info['second_order']) + '_' + str(info['id']) + '_' + info['type'] + '_' + 'cc' + '_' + info['k']
                if info['inverse']:
                    '''存其他token的cross_attention'''
                    editing_strategy = info['editing_strategy']
                    ci_ic_ii_cc_ratio = info['ci_ic_ii_cc_ratio']
                    if 'ci' in editing_strategy:
                        info['feature'][ci_feature_name] = (attn_weight[:, :, :512,512:] * ci_ic_ii_cc_ratio[0]).cpu()
                    if 'ic' in editing_strategy:
                        info['feature'][ic_feature_name] = (attn_weight[:, :, 512:,:512]* ci_ic_ii_cc_ratio[1]).cpu()
                    if 'cc' in editing_strategy:
                        info['feature'][cc_feature_name] = (attn_weight[:, :, :512,:512] * ci_ic_ii_cc_ratio[2]).cpu()
                    if 'ii' in editing_strategy:
                        info['feature'][ii_feature_name] = (attn_weight[:, :, 512:,512:]* ci_ic_ii_cc_ratio[3]).cpu()
            
                else:
                    '''替换其他token的cross_attention'''
                    editing_strategy = info['editing_strategy']
                    # 替换CI和IC注意力权重（需要分别处理两个方向的cross attention）
                    if 'replace' in editing_strategy:
                        # 替换context->image的cross attention
                        if 'ci' in editing_strategy:
                            if ci_feature_name in info['feature']:
                                ci_value = info['feature'][ci_feature_name].cuda()
                                attn_weight=replace_cross_attn_ci_regions(
                                    attn_weight=attn_weight,
                                    ci_replacements=ci_value,
                                    alphas=info['alphas'],
                                    max_seq_len=512,
                                    mapper=info['mapper']
                                    )                
                        # 替换image->context的cross attention
                        if 'ic' in editing_strategy:
                            if ic_feature_name in info['feature']:
                                ic_value = info['feature'][ic_feature_name].cuda()
                                ic_value_ = ic_value.transpose(2,3)
                                attn_weight_ = attn_weight.transpose(2,3)
                                attn_weight_=replace_cross_attn_ci_regions(
                                    attn_weight=attn_weight_,
                                    ci_replacements=ic_value_,
                                    alphas=info['alphas'],
                                    max_seq_len=512,
                                    mapper=info['mapper'])
                                attn_weight = attn_weight_.transpose(2,3)
                        # 替换image->image的caption 的self attention
                        if 'ii' in editing_strategy:
                            if ii_feature_name in info['feature']:
                                ii_value = info['feature'][ii_feature_name].cuda()
                                attn_weight[:, :, 512:, 512:] = ii_value.unsqueeze(1)
                        # 替换context->context的image 的self attention
                        if 'cc' in editing_strategy:
                            if cc_feature_name in info['feature']:
                                cc_value = info['feature'][cc_feature_name].cuda()
                                attn_weight[:, :, :512, :512] = cc_value.unsqueeze(1)
                
                    
                    # 累加CI和IC注意力权重
                    if 'add' in editing_strategy:
                        if 'ci' in editing_strategy:
                            if ci_feature_name in info['feature']:
                                ci_value = info['feature'][ci_feature_name].cuda()
                                # 扩展维度匹配头数 [batch, seq_ci, key_ci] -> [batch, num_heads, seq_ci, key_ci]
                                attn_weight[:, :, :512, 512:] += ci_value
                        # 替换image->context的cross attention
                        if 'ic' in editing_strategy:
                            if ic_feature_name in info['feature']:
                                ic_value = info['feature'][ic_feature_name].cuda()
                                attn_weight[:, :, 512:, :512] += ic_value
                        # 替换image->image的caption 的self attention
                        if 'ii' in editing_strategy:
                            if ii_feature_name in info['feature']:
                                ii_value = info['feature'][ii_feature_name].cuda()
                                attn_weight[:, :, 512:, 512:] += ii_value
                        # 替换context->context的image 的self attention
                        if 'cc' in editing_strategy:
                            if cc_feature_name in info['feature']:
                                cc_value = info['feature'][cc_feature_name].cuda()
                                attn_weight[:, :, :512,:512] += cc_value
        elif info['type_s'] == 'reweight':
            if info['inject'] and info['id'] <= info['end_layer_index'] and info['id'] >= info['start_layer_index']:
                if info['inverse'] == False:
                    inds_word = info['inds_word']
                    reweight_times = info['reweight_times']
                    attn_weight[:, :, inds_word,512:] = attn_weight[:, :, inds_word,512:] * reweight_times
                    attn_weight[:, :, 512:,inds_word] = attn_weight[:, :, 512:,inds_word] * reweight_times
            '''此处可以继续加功能'''



            
        attn = attn_weight @ v  # 矩阵乘法得到最终注意力输出  #1*24*1772*128
        attn = self.reshape(attn)

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output, info


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
    