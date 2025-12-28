import torch
import warnings
from ollm import Inference, file_get_contents, TextStreamer

# Suppress the specific warning about torch_dtype being deprecated as we are using dtype correctly
warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")

o = Inference("qwen3-next-80B", device="cuda:0", logging=True) #llama3-1B/3B/8B-chat, gpt-oss-20B, qwen3-next-80B
o.ini_model(models_dir="./models/", force_download=False)
o.offload_layers_to_cpu(layers_num=48) #(optional) offload some layers to CPU for speed boost
past_key_values = o.DiskCache(cache_dir="./kv_cache/") #set None if context is small
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

sm = "You are helpful AI assistant"
um = "List planets starting from Mercury, Show any newly found planets that are not the traditional planets"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

# Fixed: Removed reasoning_effort argument as it's not supported
input_ids = o.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(o.device)

# Fixed: Create attention_mask explicitly to avoid warning and potential issues
attention_mask = torch.ones_like(input_ids)

# Fixed: Passed attention_mask to generate
outputs = o.model.generate(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, max_new_tokens=500, streamer=text_streamer).cpu()
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
