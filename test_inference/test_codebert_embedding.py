import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
from codetf.models import load_model_pipeline

model = load_model_pipeline(model_name="bert", task="pretrained",
            model_type="codebert-base", is_eval=True,
            load_in_8bit=True, weight_sharding=False)

code_snippet = "def print_hello_world():"
embeddings = model.predict([code_snippet])

# embedding of "code_snippet"
print(code_snippets)