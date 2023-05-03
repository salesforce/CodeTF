import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from transformers import AutoTokenizer
from codetf.models.base_model import BaseModel

class GPTBaseModel(BaseModel):

    @classmethod
    def init_tokenizer(cls, model):
        return AutoTokenizer.from_pretrained(model)
    
  
   