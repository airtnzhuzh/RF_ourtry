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
def find_same_token_ids(x: str, y: str, tokenizer, max_len=512):
    """获得source prompt和target prompt对应的不同token id, 例如source 中的第1,2,3项,对应target中的第3,4项"""
    """ 其实我感觉这个函数就够用了，不需要获得映射矩阵"""
    words_x = x.split(' ')
    words_y = y.split(' ')
    
    if len(words_x) != len(words_y):
        raise ValueError(f"Prompts must have same word count. X: {len(words_x)}, Y: {len(words_y)}")
    
    # 获取需保留的单词位置
    inds_remain = [i for i in range(len(words_y)) if words_y[i] == words_x[i]]
    
    # 获取对应token索引（T5适配）
    inds_source = [get_word_inds_t5(x, i, tokenizer) for i in inds_remain]
    inds_target = [get_word_inds_t5(y, i, tokenizer) for i in inds_remain]
    
    # Flatten the inds_source list and find the maximum index
    flat_inds_source = [idx for sublist in inds_source for idx in sublist]
    if flat_inds_source:
        last_idx = max(flat_inds_source)
        # Add all indices from last_idx+1 to max_len-1
        additional_inds = list(range(last_idx + 1, max_len))
        # Add these additional indices to inds_source (as separate lists)
        for idx in additional_inds:
            inds_source.append(np.array([idx]))
            # For target, we'll add the same indices (or you might want to handle differently)
            inds_target.append(np.array([idx]))
    
    merged_source = np.concatenate(inds_source)
    merged_target = np.concatenate(inds_target)
    return [merged_source], [merged_target]

def find_word_token_ids(prompt: str, word: str, tokenizer, max_len=512):
    words_prompt = prompt.split(' ')
    
    if word not in words_prompt:
        raise ValueError(f"Word '{word}' not found in prompt")
    
    # 获取需替换的单词位置
    inds_replace = [i for i in range(len(words_prompt)) if word == words_prompt[i]]
    
    # 获取对应token索引（T5适配）
    # 逐个处理索引，避免传入列表导致错误
    inds_source = []
    for i in inds_replace:
        inds = get_word_inds_t5(prompt, i, tokenizer)
        inds_source.extend(inds.tolist())  # 合并结果
    
    return inds_source 

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
def replace_cross_attn_ci_regions_(
    attn_weight: torch.Tensor, 
    ci_replacements: torch.Tensor,  # 预存的ci区域替换张量 {step: tensor}
    max_seq_len: int,
    inds_target: np.ndarray,
    inds_source: np.ndarray):


    inds_source = np.array(inds_source)
    indices_source = torch.from_numpy(inds_source).long()
    inds_target = np.array(inds_target)
    indices_target = torch.from_numpy(inds_target).long()
    if (indices_source < 0).any() or (indices_source >= ci_replacements.shape[2]).any():
        raise ValueError(f"索引越界: attn形状为{ci_replacements.shape}, 有效索引范围0-{ci_replacements.shape[0]-1}")
    min_ind = indices_source[0]
    max_ind = indices_source[-1]
    
    # 检查indices_target和indices_source长度相同
    if (len(indices_target) == len(indices_source)) or len(indices_target) > len(indices_source):
        attn_weight[:, :, :min_ind, 512:] = ci_replacements[:, :, :min_ind, :]
        attn_weight[:, :, (max_ind + 1):max_seq_len, 512:] = ci_replacements[:, :, (max_ind + 1):, :]
    else: # len(indices_target) < len(indices_source)
                # 计算需要复制的行数，确保 n_rows 是整数
        n_rows = int(indices_source[-1] - indices_target[-1] - 1)

        # 提取源数据
        source_data = attn_weight[:, :, indices_target[-1], 512:]

        # 扩展源数据以匹配目标切片的形状
        # 在第三个维度上添加一个维度，然后扩展到 n_rows 行
        expanded_data = source_data.unsqueeze(2).expand(-1, -1, n_rows, -1)

        # 将扩展后的数据赋值给目标切片
        attn_weight[:, :, indices_target[-1]+1:indices_source[-1], 512:] = expanded_data
    

    return attn_weight


@staticmethod
def replace_cross_attn_ci_regions(
    attn_weight: torch.Tensor, 
    ci_replacements: torch.Tensor,  # 预存的ci区域替换张量 {step: tensor}
    max_seq_len: int,
    inds_target: np.ndarray,
    inds_source: np.ndarray,
    mapper : torch.Tensor
    ):


    inds_source = np.array(inds_source)
    indices_source = torch.from_numpy(inds_source).long()
    inds_target = np.array(inds_target)
    indices_target = torch.from_numpy(inds_target).long()
    
    for i in range(max_seq_len):  # 0到511
        if i in inds_source:
            j = int(np.where(mapper[i] > 0)[0][0])  # 找到对应的目标索引
            attn_weight[:, :, j, 512:] = ci_replacements[:, :, i, :]
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
    inds_target = np.array(inds_target)
    indices_target = torch.from_numpy(inds_target).long()
    if (indices_target < 0).any() or (indices_target >= ic_replacements.shape[2]).any():
        raise ValueError(f"索引越界: attn形状为{ic_replacements.shape}, 有效索引范围0-{ic_replacements.shape[0]-1}")
    min_ind = indices_target[0]
    max_ind = indices_target[-1]

    # 检查indices_target和indices_source长度相同
    if (len(indices_target) == len(indices_source)) or len(indices_target) > len(indices_source):
        attn_weight[:, :, max_seq_len:,:min_ind ] = ic_replacements[:, :, :, :min_ind]#1*24*4592*4592 1*24*4080*512
        attn_weight[:, :,max_seq_len:, (max_ind + 1):max_seq_len] = ic_replacements[:, :, :, :(max_ind + 1):max_seq_len]
    else:
                # 计算需要复制的行数
        n_rows = int(indices_source[-1] - indices_target[-1] - 1)

        # 提取源数据
        source_data = attn_weight[:, :, max_seq_len:, indices_target[-1]]

        # 扩展源数据以匹配目标切片的形状
        expanded_data = source_data.unsqueeze(-1).expand(-1, -1, -1, n_rows)

        # 将扩展后的数据赋值给目标切片
        attn_weight[:, :, max_seq_len:, indices_target[-1]+1:indices_source[-1]] = expanded_data
    return attn_weight