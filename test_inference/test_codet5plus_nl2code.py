import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from codetf.models import load_model_pipeline

code_generation_model = load_model_pipeline(model_name="codet5", task="pretrained",
            model_type="plus-2B", is_eval=True,
            load_in_8bit=False, load_in_4bit=False, weight_sharding=True)
            
result = code_generation_model.predict(["def print_hello_world():"], max_length=15)
print(result)