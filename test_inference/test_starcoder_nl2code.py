import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
from codetf.models import load_model_pipeline

model = load_model_pipeline(model_name="starcoder", task="pretrained",
                model_type="15.5B",
                quantize="int8", quantize_algo="bitsandbyte")

prompts = "def print_hello_world():"
code_snippets = model.predict([prompts])

print(code_snippets)