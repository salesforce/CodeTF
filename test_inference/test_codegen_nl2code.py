import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from codetf.models import load_model_pipeline

code_generation_model = load_model_pipeline(model_name="causallm", task="pretrained",
            model_type="codegen-2B-mono", is_eval=True,
            load_in_8bit=True, load_in_4bit=False, weight_sharding=False)
            
result = code_generation_model.predict(["# this function prints hello world"])
print(result)