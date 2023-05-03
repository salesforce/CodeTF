import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
from codetf.models import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = "cpu"
model = load_model(name="codegen_nl2code", model_type="2B-mono", is_eval=True, device=device)


prompts = "# this function prints hello world"
code_snippets = model.predict([prompts])

print(code_snippets)