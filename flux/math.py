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
def find_same_token_ids(x: str, y: str, tokenizer, max_len=512, window_size=3):
    """改进版：允许在滑动窗口范围内寻找局部匹配"""
    words_x = x.split(' ')
    words_y = y.split(' ')
    min_len = min(len(words_x), len(words_y))
    
    inds_remain = []
    i = 0
    while i < min_len:
        # 如果当前单词匹配
        if words_x[i] == words_y[i]:
            inds_remain.append(i)
            i += 1
        else:
            # 在滑动窗口范围内查找后续可能的对齐点
            found = False
            for j in range(1, window_size+1):
                if i+j >= min_len:
                    break
                if words_x[i+j] == words_y[i]:
                    # 跳过x中不匹配的部分
                    i += j
                    found = True
                    break
                elif words_x[i] == words_y[i+j]:
                    # 跳过y中不匹配的部分
                    i += j
                    found = True
                    break
            
            if not found:
                i += 1  # 当前窗口无匹配，继续向后搜索

    print("Aligned indices:", inds_remain)
    
    # 后续处理保持不变
    inds_source = [get_word_inds_t5(x, i, tokenizer) for i in inds_remain]
    inds_target = [get_word_inds_t5(y, i, tokenizer) for i in inds_remain]
    print("inds_source:", inds_source)
    print("inds_target:", inds_target)
    
    # 处理空列表的情况
    if not inds_source:
        return [np.array([], dtype=int)], [np.array([], dtype=int)]
    
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
    
    print("=="*20)
    print("inds_source:", inds_source)
    print("inds_target:", inds_target)
    
    # 处理可能的空数组情况
    try:
        merged_source = np.concatenate(inds_source) if inds_source else np.array([], dtype=int)
        merged_target = np.concatenate(inds_target) if inds_target else np.array([], dtype=int)
    except ValueError:
        merged_source = np.array([], dtype=int)
        merged_target = np.array([], dtype=int)
        
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
    
    # if len(words_x) != len(words_y):
    #     raise ValueError(f"Prompts must have same word count. X: {len(words_x)}, Y: {len(words_y)}")
    # 获取两个句子中较短的长度
    min_len = min(len(words_x), len(words_y))
    

    # 获取需替换的单词位置
    inds_replace = [i for i in range(min_len) if words_y[i] != words_x[i]]
    
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
    alphas: torch.Tensor,
    mapper : torch.Tensor
    ):
    for i in range(max_seq_len):
        if alphas[i] == 1:
            attn_weight[:, :, mapper[i], 512:] = ci_replacements[:, :, i, :]# replacements是source的, attn_weight是target的    


    return attn_weight

# @staticmethod
# def replace_cross_attn_ic_regions(
#     attn_weight: torch.Tensor, 
#     ic_replacements: torch.Tensor,  # 预存的ic区域替换张量 {step: tensor}
#     max_seq_len: int,
#     inds_target: np.ndarray,
#     inds_source: np.ndarray):

#     inds_source = np.array(inds_source)
#     indices_source = torch.from_numpy(inds_source).long()
#     inds_target = np.array(inds_target)
#     indices_target = torch.from_numpy(inds_target).long()
#     if (indices_target < 0).any() or (indices_target >= ic_replacements.shape[2]).any():
#         raise ValueError(f"索引越界: attn形状为{ic_replacements.shape}, 有效索引范围0-{ic_replacements.shape[0]-1}")
#     min_ind = indices_target[0]
#     max_ind = indices_target[-1]

#     # 检查indices_target和indices_source长度相同
#     if (len(indices_target) == len(indices_source)) or len(indices_target) > len(indices_source):
#         attn_weight[:, :, max_seq_len:,:min_ind ] = ic_replacements[:, :, :, :min_ind]#1*24*4592*4592 1*24*4080*512
#         attn_weight[:, :,max_seq_len:, (max_ind + 1):max_seq_len] = ic_replacements[:, :, :, :(max_ind + 1):max_seq_len]
#     else:
#                 # 计算需要复制的行数
#         n_rows = int(indices_source[-1] - indices_target[-1] - 1)

#         # 提取源数据
#         source_data = attn_weight[:, :, max_seq_len:, indices_target[-1]]

#         # 扩展源数据以匹配目标切片的形状
#         expanded_data = source_data.unsqueeze(-1).expand(-1, -1, -1, n_rows)

#         # 将扩展后的数据赋值给目标切片
#         attn_weight[:, :, max_seq_len:, indices_target[-1]+1:indices_source[-1]] = expanded_data
#     return attn_weight


class ScoreParams:

    def __init__(self, gap, match, mismatch):
        self.gap = gap
        self.match = match
        self.mismatch = mismatch

    def mis_match_char(self, x, y):
        if x != y:
            return self.mismatch
        else:
            return self.match
        
    
def get_matrix(size_x, size_y, gap):
    matrix = []
    for i in range(len(size_x) + 1):
        sub_matrix = []
        for j in range(len(size_y) + 1):
            sub_matrix.append(0)
        matrix.append(sub_matrix)
    for j in range(1, len(size_y) + 1):
        matrix[0][j] = j*gap
    for i in range(1, len(size_x) + 1):
        matrix[i][0] = i*gap
    return matrix


def get_matrix(size_x, size_y, gap):
    matrix = np.zeros((size_x + 1, size_y + 1), dtype=np.int32)
    matrix[0, 1:] = (np.arange(size_y) + 1) * gap
    matrix[1:, 0] = (np.arange(size_x) + 1) * gap
    return matrix


def get_traceback_matrix(size_x, size_y):
    matrix = np.zeros((size_x + 1, size_y +1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix


def global_align(x, y, score):
    matrix = get_matrix(len(x), len(y), score.gap)
    trace_back = get_traceback_matrix(len(x), len(y))
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1] + score.gap
            up = matrix[i - 1, j] + score.gap
            diag = matrix[i - 1, j - 1] + score.mis_match_char(x[i - 1], y[j - 1])
            matrix[i, j] = max(left, up, diag)
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back


def get_aligned_sequences(x, y, trace_back):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i-1])
            y_seq.append(y[j-1])
            i = i-1
            j = j-1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append('-')
            y_seq.append(y[j-1])
            j = j-1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i-1])
            y_seq.append('-')
            i = i-1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)


def get_mapper_diff_num_of_words(x: str, y: str, tokenizer, max_len=512):
    x_seq = tokenizer.encode(x)
    y_seq = tokenizer.encode(y)
    score = ScoreParams(0, 1, -1)
    matrix, trace_back = global_align(x_seq, y_seq, score)
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    alphas = torch.ones(max_len) # 0需要替换，1不需要替换
    alphas[: mapper_base.shape[0]] = mapper_base[:, 1].ne(-1).float()
    mapper = torch.zeros(max_len, dtype=torch.int64)
    mapper[:mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0]:] = len(y_seq) + torch.arange(max_len - len(y_seq))
    return mapper, alphas



