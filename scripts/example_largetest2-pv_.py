import sys
import os
import torch
import warnings

# Suppress the deprecation warning for torch_dtype
warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ollm import Inference, file_get_contents, TextStreamer

# Initialize Inference
o = Inference("qwen3-next-80B", device="cuda:0", logging=True)

# Download/Initialize model
# force_download=False means it won't re-download if the directory exists
o.ini_model(models_dir="./models/", force_download=False)

# Offload layers to CPU
# This offloads model weights *before* inference starts to save VRAM.
# The log output appearing before prompts is expected behavior as this is a setup step.
o.offload_layers_to_cpu(layers_num=48)

# Initialize DiskCache for KV cache
past_key_values = o.DiskCache(cache_dir="./kv_cache/")

# Initialize TextStreamer
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

# Prepare messages
sm = "You are helpful AI assistant"
um = "List planets starting from Mercury, Show any newly found planets that are not the traditional planets"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

# Apply chat template
# 'reasoning_effort' is removed as it's not supported.
# We manually handle the input_ids tensor.
input_ids = o.tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(o.device)

# Create attention mask manually since batch size is 1 and pad == eos might confuse automatic inference
attention_mask = torch.ones_like(input_ids)

# Generate response
outputs = o.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask, # Explicitly pass attention mask
    past_key_values=past_key_values,
    max_new_tokens=500,
    streamer=text_streamer
).cpu()

# Decode output (the streamer prints it as it goes, but we can also decode the full output if needed)
# Since we used a streamer, we might just want to print a separator or nothing.
# But the original script decoded and printed the answer again.
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print("\nFinal Answer:\n" + answer)
