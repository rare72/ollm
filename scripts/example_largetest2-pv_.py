
import sys, os
import torch
sys.path.append(os.path.abspath("src"))
from ollm import Inference, TextStreamer

# Initialize the model
# User note: "flash_attention_2 is not imported" warning is expected if flash_attn is not installed.
# The code handles fallback gracefully.
o = Inference("qwen3-next-80B", device="cuda:0", logging=True)

# Load the model
# User note: `torch_dtype` warning is from transformers internals; this script uses `dtype` correctly in the library.
o.ini_model(models_dir="./models/", force_download=False)

# Offload layers to CPU
# Logic explanation: This moves model WEIGHTS to CPU to save GPU VRAM before inference starts.
# It does NOT involve the prompt or KV cache yet.
o.offload_layers_to_cpu(layers_num=48)

# Initialize KV Cache
past_key_values = o.DiskCache(cache_dir="./kv_cache/")

# Setup streamer
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

# Prepare inputs
sm = "You are helpful AI assistant"
um = "List planets starting from Mercury, Show any newly found planets that are not the traditional planets"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

# Removed 'reasoning_effort="minimal"' as it is not supported/needed.
input_ids = o.tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(o.device)

# Create attention mask (required for batch size 1 when pad_token_id == eos_token_id)
# This prevents "The attention mask is not set..." warning/error.
attention_mask = torch.ones_like(input_ids)

# Generate
outputs = o.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    past_key_values=past_key_values,
    max_new_tokens=500,
    streamer=text_streamer
).cpu()

# Decode and print answer
# Note: TextStreamer already prints to stdout, but we decode here for the final variable 'answer'.
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print("\nFinal Answer:\n", answer)
