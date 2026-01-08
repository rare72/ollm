import codecs, io, time
import torch 

def file_put_contents(filename, st):
    file = codecs.open(filename, "w", "utf-8")
    file.write(st)
    file.close()

def file_get_contents(name):
    f = io.open(name, mode="r", encoding="utf-8") #utf-8 | Windows-1252
    return f.read()
    
def tensor_size_gb(t: torch.Tensor) -> float:
    return t.numel() * t.element_size() / 1024**3

class Stats:
    def __init__(self):
        self.d = {}

    def set(self, name, t1):
        if name not in self.d: self.d[name] = []
        self.d[name].append( round(time.perf_counter() - t1, 3) ) 

    def print_and_clean(self):
        st = "Stats:"
        for name, a in self.d.items():
            st+=f" {name}: {a[:5]} t:{round(sum(a), 3)},"
        self.d = {}
        return st


# === Helper utilities ===
def _walk_to_parent(obj, attr_path):
    """Return (parent_obj, leaf_name) for attr_path like 'self_attn.q_proj.weight'"""
    parts = attr_path.split('.')
    parent = obj
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def _assign_tensor_to_module(target_parent, leaf, tensor):
    """
    Assign a tensor into target_parent.<leaf>.
    - If target_parent.<leaf> has a .load call, call it with tensor.
    - Else, if attribute endswith 'weight' or 'bias' and current attr is nn.Parameter, replace it.
    - Else, set attribute to nn.Parameter(tensor) (read-only).
    """
    existing = getattr(target_parent, leaf, None)

    # If target object has a load(tensor) method (user's custom modules), call it.
    if hasattr(existing, "load") and callable(getattr(existing, "load")):
        existing.load(tensor)   # user-supplied API
        return

    # If existing is a Parameter (typical), replace with new Parameter on CUDA
    if isinstance(existing, torch.nn.Parameter) or getattr(existing, "__class__", None) is torch.nn.Parameter:
        param = torch.nn.Parameter(tensor.detach(), requires_grad=False)
        setattr(target_parent, leaf, param)
        return

    # If attribute is a module (like a Linear) we attempt to set its weight/bias
    if isinstance(existing, torch.nn.Linear) or hasattr(existing, "weight"):
        # try to set weight and bias if given tensor is 2D weight
        if tensor.ndim == 2 and hasattr(existing, "weight"):
            existing.weight = torch.nn.Parameter(tensor.detach(), requires_grad=False)
            return
        # fallback: set attribute to Parameter
    # Default fallback: replace attribute with a Parameter
    setattr(target_parent, leaf, torch.nn.Parameter(tensor.detach(), requires_grad=False))


def _set_meta_placeholder(target_parent, leaf):
    """Replace parameter/module attribute with a tiny meta-device Parameter to free VRAM."""
    placeholder = torch.nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)
    setattr(target_parent, leaf, placeholder)


def remove_layers_weights(model):
    # 2. Remove heavy decoder block weights (keep skeleton)
    for layer in model.model.layers:
        for name, module in layer.named_children():
            if hasattr(module, "weight"):
                module.weight = torch.nn.Parameter(
                    torch.empty(0), requires_grad=False
                )                
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = torch.nn.Parameter(
                    torch.empty(0), requires_grad=False
                )



class ScriptHelper:
    def __init__(self, inference_instance=None):
        self.inference = inference_instance
        self.start_time = None
        self.end_time = None

    def init(self):
        self.start_time = time.time()
        print(f"\n[INIT] Script started at {time.ctime(self.start_time)}")

        # Check hidden_act if model is available (or will be)
        # Since this might be called before model load, we might need to call it again or check later.
        # But user wants an [INIT] section output.
        if self.inference and hasattr(self.inference, 'model') and hasattr(self.inference.model, 'config'):
            self._print_config(self.inference.model.config)

    def print_config(self, config):
        """Helper to print config details if they weren't ready at init"""
        print(f"[INIT] Configuration loaded.")
        if hasattr(config, 'hidden_act'):
            print(f"[INIT] Model hidden_act: {config.hidden_act}")
        elif hasattr(config, 'activation_function'):
            print(f"[INIT] Model activation_function: {config.activation_function}")

    def exit(self, tokenizer, input_ids, output_ids, final_answer):
        self.end_time = time.time()
        print(f"\n[EXIT] Script finished at {time.ctime(self.end_time)}")
        if self.start_time:
            duration = self.end_time - self.start_time
            print(f"[EXIT] Duration: {duration:.2f} seconds")

        # Calculate counts
        input_count = input_ids.shape[1] if hasattr(input_ids, 'shape') else len(input_ids)

        # Handle output_ids potentially being a list or tensor
        if hasattr(output_ids, 'shape'):
            total_tokens = output_ids.shape[1]
        else:
            total_tokens = len(output_ids)

        # Calculate readable tokens from the final answer string
        readable_count = 0
        if tokenizer:
             # encode without special tokens to get pure content count
             readable_ids = tokenizer.encode(final_answer, add_special_tokens=False)
             readable_count = len(readable_ids)

        print(f"[EXIT] Total Input Prompt Tokens: {input_count}")
        print(f"[EXIT] Total Context Tokens (Prompt + Generated): {total_tokens}")
        print(f"[EXIT] Total Readable Tokens (Answer content): {readable_count}")

        print("\nFinal Answer:\n", final_answer)
