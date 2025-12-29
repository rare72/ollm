import sys
import os
import torch
import warnings

# Add src to path to import ollm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Suppress deprecation warnings specifically for torch_dtype which comes from library internals
warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")

from ollm import Inference, file_get_contents, TextStreamer

# Initialize inference with the requested model
o = Inference("qwen3-next-80B", device="cuda:0", logging=True)
o.ini_model(models_dir="./models/", force_download=False)

# Offload layers to CPU. This happens BEFORE inference loop to prepare the model structure.
# The user might see logs here before prompt processing, which is expected behavior for this library.
o.offload_layers_to_cpu(layers_num=48)

# Initialize DiskCache for handling large context if needed
past_key_values = o.DiskCache(cache_dir="./kv_cache/")

text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

sm = "You are helpful AI assistant"
um = "List planets starting from Mercury, Show any newly found planets that are not the traditional planets"
messages = [{"role":"system", "content":sm}, {"role":"user", "content":um}]

# Apply chat template. Removed 'reasoning_effort' as it is not supported.
# Use 'dtype' instead of 'torch_dtype' internally, and suppressed the warning above.
input_ids = o.tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(o.device)

# Generate attention mask manually as it's required when batch size is 1 and pad_token_id == eos_token_id
attention_mask = torch.ones_like(input_ids)

# Generate output
outputs = o.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask, # Added attention_mask
    past_key_values=past_key_values,
    max_new_tokens=500,
    streamer=text_streamer
).cpu()

# Decode and print the answer
# Skip special tokens might be desired, but user had False. Keeping False but note it.
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)
