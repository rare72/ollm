
import sys
import os
import torch
from transformers import AutoTokenizer
from transformers import PretrainedConfig

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from ollm.configuration_deepseek import DeepseekConfig

model_dir = "./models/deepseek-moe"
if not os.path.exists(model_dir):
    print(f"Model directory {model_dir} not found, using default ID")
    model_id = "deepseek-ai/deepseek-moe-16b-chat"
else:
    model_id = model_dir

print(f"Loading tokenizer from {model_id}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"Tokenizer loaded.")
    print(f"Tokenizer vocab size (len): {len(tokenizer)}")
    print(f"Tokenizer vocab_size attribute: {tokenizer.vocab_size}")

    # Check config
    config = DeepseekConfig()
    print(f"Default DeepseekConfig vocab_size: {config.vocab_size}")

    # Try loading actual config if exists
    try:
        from transformers import AutoConfig
        actual_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"Actual Model Config vocab_size: {actual_config.vocab_size}")

        if len(tokenizer) > actual_config.vocab_size:
            print(f"MISMATCH DETECTED: Tokenizer length ({len(tokenizer)}) > Config vocab_size ({actual_config.vocab_size})")
        else:
            print("No mismatch detected.")

    except Exception as e:
        print(f"Could not load actual config: {e}")

except Exception as e:
    print(f"Error loading tokenizer: {e}")
