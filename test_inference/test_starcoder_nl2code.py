import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
from codetf.models import load_model_pipeline

model = load_model_pipeline(model_name="causal-lm", task="pretrained",
            model_type="starcoder-15.5B", is_eval=True,
            load_in_8bit=True, weight_sharding=False)

prompts = "def print_hello_world():"
code_snippets = model.predict([prompts])

print(code_snippets)