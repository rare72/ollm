import warnings
# Filter warnings about torch_dtype being deprecated, as this might be coming from internal library calls
warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")

import torch
from ollm import Inference, file_get_contents, TextStreamer

# Initialize inference with the model
o = Inference("qwen3-next-80B", device="cuda:0", logging=True)
o.ini_model(models_dir="./models/", force_download=False)

# Offload layers to CPU (pre-loading step)
o.offload_layers_to_cpu(layers_num=48)

# Initialize DiskCache
past_key_values = o.DiskCache(cache_dir="./kv_cache/")

# Initialize TextStreamer
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

sm = "You are helpful AI assistant"
um = "List planets starting from Mercury, Show any newly found planets that are not the traditional planets"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

# Apply chat template
# Removed 'reasoning_effort' as it is not supported
input_ids = o.tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(o.device)

# Create attention mask explicitly
# This prevents the warning: "The attention mask is not set and cannot be inferred..."
attention_mask = torch.ones_like(input_ids)

# Generate response
outputs = o.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask, # Pass the attention mask
    past_key_values=past_key_values,
    max_new_tokens=500,
    streamer=text_streamer
).cpu()

# Decode and print answer
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
