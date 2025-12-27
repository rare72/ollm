
import sys
import os
import torch

# Add src to pythonpath so ollm can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ollm import Inference, file_get_contents, TextStreamer

# 1. Initialize Inference
# Fix: Ensure device is passed as a string (e.g., "cuda:0").
# Fix: 'logging=True' is fine.
o = Inference("qwen3-next-80B", device="cuda:0", logging=True)

# 2. Initialize Model
# Fix: Using 'force_download=False' to avoid re-downloading if already present.
# Note: This step loads the model structure.
o.ini_model(models_dir="./models/", force_download=False)

# 3. Offload Layers
# Fix/Clarification: This offloads model weights to CPU/GPU *before* prompt processing.
# The user log "offloading layers to CPU... but the message or prompt has not been loaded yet" is expected behavior.
o.offload_layers_to_cpu(layers_num=48)

# 4. Setup KV Cache
# Fix: Using 'DiskCache' is correct for large context/models if needed.
past_key_values = o.DiskCache(cache_dir="./kv_cache/")

# 5. Setup Streamer
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

# 6. Prepare Messages
sm = "You are helpful AI assistant"
um = "List planets starting from Mercury, Show any newly found planets that are not the traditional planets"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

# 7. Tokenize
# Fix: Removed 'reasoning_effort="minimal"' as it is not supported.
# Fix: 'apply_chat_template' with 'return_tensors="pt"' returns a tensor of input_ids.
input_ids = o.tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(o.device)

# 8. Create Attention Mask
# Fix: Explicitly create attention_mask.
# "The attention mask is not set and cannot be inferred from input because pad token is same as eos token."
attention_mask = torch.ones_like(input_ids)

# 9. Generate
# Fix: Pass 'attention_mask' to 'generate'.
outputs = o.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    max_new_tokens=500,
    streamer=text_streamer
).cpu()

# 10. Decode and Print
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
