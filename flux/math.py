import torch
from einops import rearrange
from torch import Tensor
import numpy as np

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

@staticmethod
def find_diff_token_ids(x: str, y: str, tokenizer, max_len=512):
    """获得source prompt和target prompt对应的不同token id, 例如source 中的第1,2,3项,对应target中的第3,4项"""
    """ 其实我感觉这个函数就够用了，不需要获得映射矩阵"""
    words_x = x.split(' ')
    words_y = y.split(' ')
    
    if len(words_x) != len(words_y):
        raise ValueError(f"Prompts must have same word count. X: {len(words_x)}, Y: {len(words_y)}")
    
    # 获取需替换的单词位置
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    
    # 获取对应token索引（T5适配）
    inds_source = [get_word_inds_t5(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds_t5(y, i, tokenizer) for i in inds_replace]
    
    return inds_source, inds_target

@staticmethod
def get_mapper(x: str, y: str, tokenizer, max_len=512):
    """针对T5的映射矩阵生成,处理句子级对齐与子词差异"""
    words_x = x.split(' ')
    words_y = y.split(' ')
    
    if len(words_x) != len(words_y):
        raise ValueError(f"Prompts must have same word count. X: {len(words_x)}, Y: {len(words_y)}")
    
    # 获取需替换的单词位置
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    
    # 获取对应token索引（T5适配）
    inds_source = [get_word_inds_t5(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds_t5(y, i, tokenizer) for i in inds_replace]
    
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    # 动态对齐源与目标的token位置
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and i == inds_source[cur_inds][0]:
            src = inds_source[cur_inds]
            tgt = inds_target[cur_inds]
            
            # 处理子词长度不匹配（按比例分配注意力）
            if len(src) == len(tgt):
                mapper[src, tgt] = 1
            else:
                ratio = 1 / len(tgt)
                for t in tgt:
                    mapper[src, t] = ratio
            i += len(src)
            j += len(tgt)
            cur_inds += 1
        else:
            # 非替换区域保持对角映射
            mapper[i, j] = 1
            i += 1
            j += 1
    return torch.tensor(mapper, dtype=torch.float32)

def get_word_inds_t5(text: str, word_place: int, tokenizer):
    """T5分词器适配版本,处理子词前缀'▁'并定位目标单词的token索引"""
    split_text = text.split(" ")
    if isinstance(word_place, str):
       target_indices = [i for i, word in enumerate(split_text) if word == word_place]
    elif isinstance(word_place, int):
        target_indices = [word_place]
    out = []
    if len(target_indices) > 0:
        # T5分词后保留前缀'▁'，解码时需处理
        tokens = tokenizer.encode(text, add_special_tokens=False)
        words_encode = [tokenizer.decode([t], clean_up_tokenization_spaces=False).lstrip('▁') for t in tokens]
        
        ptr, cur_len = 0, 0
        for i in range(len(words_encode)):
            # 累计字符长度（需考虑T5可能的分词差异）
            cur_len += len(words_encode[i])
            # 检查当前原始单词是否为目标位置
            if ptr in target_indices:
                out.append(i)  # T5无[CLS]/[SEP]，无需+1
            # 当累计长度超过当前单词长度，移动到下一单词
            if cur_len >= len(split_text[ptr].replace('▁', '')):  # 去除T5的前缀对比
                ptr += 1
                cur_len = 0
    return np.array(out)

@staticmethod
def replace_cross_attn_ci_regions(
    attn_weight: torch.Tensor, 
    ci_replacements: torch.Tensor,  # 预存的ci区域替换张量 {step: tensor}
    max_seq_len: int,
    inds_target: np.ndarray,
    inds_source: np.ndarray):


    inds_source = np.array(inds_source)
    indices_source = torch.from_numpy(inds_source).long()
    ci_replacements =ci_replacements.mean(dim=1)  # TODO
    if (indices_source < 0).any() or (indices_source >= ci_replacements.shape[1]).any():
        raise ValueError(f"索引越界: attn形状为{ci_replacements.shape}, 有效索引范围0-{ci_replacements.shape[0]-1}")
    ci_replacements = ci_replacements[inds_source]  
    

    # 获取当前步骤的替换数据
    ci_tensor = ci_replacements
    if ci_tensor is not None:
        # --- 处理ci区域（替换行）---
        # 切片获取ci区域 [batch, heads, max_seq_len, -1]
        ci_region = attn_weight[:, :, :max_seq_len, max_seq_len:]
        batch, heads, ci_rows, ci_cols = ci_region.shape
        
        
        # 检查替换索引有效性
        inds_target = np.array(inds_target)
        indices_target = torch.from_numpy(inds_target).long()
        if (indices_target < 0).any() or (indices_target >= ci_rows).any():
            raise ValueError(f"CI索引越界: 最大行数{ci_rows}，非法索引{indices_target[indices_target >= ci_rows]}")
            
        # 检查替换数据维度 [batch, n_indices_target, ci_cols]
        if ci_tensor.shape != (batch, len(indices_target), ci_cols):
            raise RuntimeError(f"CI替换数据维度不匹配,应为{(batch, len(indices_target), ci_cols)}，实际{ci_tensor.shape}")
        
        #如果ci_tensor行数和indices_target的数量，那么就替换
        #如果ci_tensor行数大于indices_target的数量，那么就取平均后复制indices_target的数量
        if len(indices_target) < ci_tensor.shape[1]:
            ci_tensor = ci_tensor.mean(dim=1).unsqueeze(1)
        #如果ci_tensor行数小于indices的数量，那么就剩余的indices_target数量用ci_tensor的最后一行补齐
        if len(indices_target) > ci_tensor.shape[1]:
            ci_tensor = torch.cat([ci_tensor, ci_tensor[-1].unsqueeze(0).repeat(len(indices_target)-ci_tensor.shape[1], 1)], dim=0)
         
       # 执行替换（广播到所有head）
        ci_tensor[:, indices_target, :] = ci_region.unsqueeze(1)  # [batch, heads, n_indices, cols]
        '''
        Exception has occurred: IndexError
too many indices for tensor of dimension 3
  File "/home/zhuzh/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/src/flux/math.py", line 163, in replace_cross_attn_ci_regions
    ci_tensor[:, :, indices_target, :] = ci_region.unsqueeze(1)  # [batch, heads, n_indices, cols]
  File "/home/zhuzh/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/src/flux/modules/layers.py", line 355, in forward
    attn_weight = replace_cross_attn_ci_regions(
  File "/home/zhuzh/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/src/flux/model.py", line 112, in forward
    img, info = block(img, vec=vec, pe=pe, info=info)
  File "/home/zhuzh/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/src/flux/sampling.py", line 361, in denoise_rf_zhuzh
    pred_mid, info = model(
  File "/home/zhuzh/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/src/edit.py", line 197, in main
    x, _ = denoise_strategy(model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info)
  File "/home/zhuzh/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/src/edit.py", line 298, in <module>
    main(args)
IndexError: too many indices for tensor of dimension 3
'''
        attn_weight[:, :, :max_seq_len, max_seq_len:] = ci_tensor

    return attn_weight


@staticmethod
def replace_cross_attn_ic_regions(
    attn_weight: torch.Tensor, 
    ic_replacements: torch.Tensor,  # 预存的ic区域替换张量 {step: tensor}
    max_seq_len: int,
    inds_target: np.ndarray,
    inds_source: np.ndarray):


    inds_source = np.array(inds_source)
    
    indices_source = torch.from_numpy(inds_source).long() 
    ic_replacements = ic_replacements.mean(dim=1)   
    if (indices_source < 0).any() or (indices_source >= ic_replacements.shape[1]).any():
        raise ValueError(f"索引越界: attn形状为{ic_replacements.shape}, 有效索引范围0-{ic_replacements.shape[0]-1}")
    ic_replacements = ic_replacements[inds_source]  
    

    # 获取当前步骤的替换数据
    ic_tensor = ic_replacements
    if  ic_tensor is not None:
        # --- 处理ic区域（替换行）---
        # 切片获取ic区域 [batch, heads, max_seq_len, -1]
        ic_region = attn_weight[:, :, :max_seq_len, max_seq_len:]
        batch, heads, ic_rows, ic_cols = ic_region.shape
        
        
        # 检查替换索引有效性
        inds_target = np.array(inds_target)
        indices_target = torch.from_numpy(inds_target).long()
        if (indices_target < 0).any() or (indices_target >= ic_rows).any():
            raise ValueError(f"CI索引越界: 最大行数{ic_rows}，非法索引{indices_target[indices_target >= ic_rows]}")
            
        # 检查替换数据维度 [batch, n_indices_target, ic_cols]
        if ic_tensor.shape != (batch, len(indices_target), ic_cols):
            raise RuntimeError(f"CI替换数据维度不匹配,应为{(batch, len(indices_target), ic_cols)}，实际{ic_tensor.shape}")
        
        #如果ic_tensor行数和indices_target的数量，那么就替换
        #如果ic_tensor行数大于indices_target的数量，那么就取平均后复制indices_target的数量
        if len(indices_target) < ic_tensor.shape[1]:
            ic_tensor = ic_tensor.mean(dim=1).unsqueeze(1)
        #如果ic_tensor行数小于indices的数量，那么就剩余的indices_target数量用ci_tensor的最后一行补齐
        if len(indices_target) > ic_tensor.shape[1]:
            ic_tensor = torch.cat([ic_tensor, ic_tensor[-1].unsqueeze(0).repeat(len(indices_target)-ic_tensor.shape[1], 1)], dim=0)
         
       # 执行替换（广播到所有head）
        ic_tensor[:, :, indices_target, :] = ic_region.unsqueeze(1)  # [batch, heads, n_indices, cols]
        attn_weight[:, :, :max_seq_len, max_seq_len:] = ic_tensor

    return attn_weight