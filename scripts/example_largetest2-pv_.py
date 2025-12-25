from ollm import Inference, file_get_contents, TextStreamer
import torch

# Initialize inference
o = Inference("qwen3-next-80B", device="cuda:0", logging=True) #llama3-1B/3B/8B-chat, gpt-oss-20B, qwen3-next-80B
o.ini_model(models_dir="./models/", force_download=False)

# Offload layers to CPU (pre-inference setup)
# This prepares the model layers in memory (RAM vs VRAM) before any prompt processing.
o.offload_layers_to_cpu(layers_num=48)

past_key_values = o.DiskCache(cache_dir="./kv_cache/") #set None if context is small
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

sm = "You are helpful AI assistant"
um = "List planets starting from Mercury, Show any newly found planets that are not the traditional planets"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

# Fixed: removed reasoning_effort="minimal" as it is not supported
# Fixed: apply_chat_template returns a tensor (return_tensors="pt"), so we use it directly
input_ids = o.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(o.device)

# Fixed: Explicitly create attention_mask as pad_token_id == eos_token_id
attention_mask = torch.ones_like(input_ids).to(o.device)

# Generate response
# Fixed: Pass attention_mask
outputs = o.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    max_new_tokens=500,
    streamer=text_streamer
).cpu()

# Decode and print answer
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
