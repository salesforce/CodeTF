import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from codetf.models.base_model import BaseModel
from codetf.common.registry import registry
from collections import defaultdict
from tqdm import tqdm
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download
import torch

@registry.register_model("causallm")
class CausalLMModel(BaseModel):

    MODEL_DICT = "configs/inference/causal_lm.yaml"
    

    def __init__(self, model, model_config, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_prediction_length = model_config["max_prediction_length"]

    @classmethod
    def init_tokenizer(cls, model):
        tokenizer = AutoTokenizer.from_pretrained(model)
        return tokenizer
    
    @classmethod
    def load_model_from_config(model_class, model_config, load_in_8bit=False, load_in_4bit=False, weight_sharding=False):
        checkpoint = model_config["huggingface_url"]

        if load_in_8bit and load_in_4bit:
            raise ValueError("Only one of load_in_8bit or load_in_4bit can be True. Please choose one.")
        
        if weight_sharding:
            weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")
            config = AutoConfig.from_pretrained(checkpoint)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)

            model.tie_weights()            
            model = load_checkpoint_and_dispatch(
                model, weights_location, device_map="auto", no_split_module_classes=["GPTJBlock"]
            )
        else:
            if load_in_8bit:
                model = AutoModelForCausalLM.from_pretrained(checkpoint, 
                                            load_in_8bit=load_in_8bit, 
                                            device_map="auto")
            elif load_in_4bit:
                model = AutoModelForCausalLM.from_pretrained(checkpoint, 
                                            load_in_4bit=load_in_4bit, 
                                            device_map="auto")
            else:
                model = AutoModelForCausalLM.from_pretrained(checkpoint, 
                                            device_map="auto")


        tokenizer = model_class.init_tokenizer(model_config["tokenizer_url"])
        
        return model_class(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer
        )
   
    def forward(self, sources):
        encoding = self.tokenizer(sources, return_tensors='pt')
        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)
        generated_ids = self.model.generate(input_ids, attention_mask=attention_mask, 
                                            max_length=self.max_prediction_length)

        predictions = self.tokenizer.batch_decode(generated_ids, truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
        return predictions

    def predict(self, sources):
        input_for_net = [' '.join(source.strip().split()).replace('\n', ' ') for source in sources]
        output = self.forward(input_for_net)
        return output