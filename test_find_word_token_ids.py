from flux.math import find_word_token_ids
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import torch
torch_device = torch.device("cuda")
prompt = "the cloudy boy is handsome"
word = "is"
tokenizer = T5TokenizerFast.from_pretrained("/home/zhuzh/t5-v1_1-xxl/")

word_id = find_word_token_ids(prompt, word, tokenizer)
print(word_id)
#成功运行