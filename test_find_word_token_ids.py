from flux.math import find_word_token_ids,get_mapper,find_same_token_ids,get_mapper_diff_num_of_words
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import torch
torch_device = torch.device("cuda")
tokenizer = T5TokenizerFast.from_pretrained("/home/zhuzh/t5-v1_1-xxl/")
prompt1 = "white flowers on a tree branch with blue sky background"
prompt2 = "an oil painting of white flowers on a tree branch with blue sky background"
prompt3 = "the 2020 honda hrx is driving down the road"
prompt4 = "the 2020 honda hrx is driving down the road full of flowers"
mapper = get_mapper(prompt1, prompt2, tokenizer, 9)
print("mapper:",mapper)
tokens_prompt1 = tokenizer(prompt3, return_tensors="pt").input_ids
tokens_prompt2 = tokenizer(prompt4, return_tensors="pt").input_ids

print("tokens_prompt1",tokens_prompt1)
print("tokens_prompt2",tokens_prompt2)
mapper,alphas = get_mapper_diff_num_of_words(prompt3, prompt4, tokenizer, 20)
print("mapper:",mapper)
print("alphas:",alphas)