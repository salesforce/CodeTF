import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from transformers import AutoTokenizer
from codetf.models.base_model import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from codetf.common.registry import registry
from accelerate import Accelerator
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download
import torch 

@registry.register_model("codet5")
class Seq2SeqModel(BaseModel):
    
    MODEL_DICT = "configs/inference/codet5.yaml"
     
    def __init__(self, model, model_config, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_source_length = model_config["max_source_length"]
        self.max_prediction_length = model_config["max_prediction_length"]
        self.beam_size = model_config["beam_size"]

    @classmethod
    def init_tokenizer(cls, model):
        return AutoTokenizer.from_pretrained(model)
    
  
    @classmethod
    def load_huggingface_model_from_config(model_class, model_config, load_in_8bit=False, load_in_4bit=False, weight_sharding=False):
        
        checkpoint = model_config["huggingface_url"]

        if load_in_8bit and load_in_4bit:
            raise ValueError("Only one of load_in_8bit or load_in_4bit can be True. Please choose one.")

        # This "device" is for the case of CodeT5plus, will be removed in the future
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if weight_sharding:
            try:
                # Try to download and load the json index file
                weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")
            except Exception:
                try:
                    # If that fails, try to download and load the bin file
                    weights_location = hf_hub_download(checkpoint, "pytorch_model.bin.index.json")
                except Exception as e:
                    # If both fail, raise an error
                    raise Exception(f"Failed to download weights: {str(e)}")
            config = AutoConfig.from_pretrained(checkpoint)
            with init_empty_weights():
                model = AutoModelForSeq2SeqLM.from_config(config)

            model.tie_weights()            
            model = load_checkpoint_and_dispatch(
                model, weights_location, model_config["device_map"], 
                no_split_module_classes=["GPTJBlock"]
            )
        else:
            if load_in_8bit:
                if model_config["device_map"]:
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, 
                                                load_in_8bit=load_in_8bit, 
                                                low_cpu_mem_usage=True,
                                                device_map="auto", trust_remote_code=model_config["trust_remote_code"])
                else: 
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, 
                                                load_in_8bit=load_in_8bit, 
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=model_config["trust_remote_code"])
            elif load_in_4bit:
                if model_config["device_map"]:
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, 
                                                load_in_4bit=load_in_4bit, 
                                                low_cpu_mem_usage=True,
                                                device_map="auto", trust_remote_code=model_config["trust_remote_code"])
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, 
                                                load_in_4bit=load_in_4bit, 
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=model_config["trust_remote_code"])
            else:
                if model_config["device_map"]:
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, 
                                                low_cpu_mem_usage=True,
                                                device_map=model_config["device_map"], trust_remote_code=model_config["trust_remote_code"])
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, 
                                                low_cpu_mem_usage=True,
                                                trust_remote_code=model_config["trust_remote_code"]).to(device)

           
        tokenizer = model_class.init_tokenizer(model_config.get("tokenizer_url"))

        return model_class(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer
        )
    
    @classmethod
    def load_custom_model(model_class, checkpoint_path, tokenizer_path, load_in_8bit=False, load_in_4bit=False):

        if load_in_8bit and load_in_4bit:
            raise ValueError("Only one of load_in_8bit or load_in_4bit can be True. Please choose one.")
        
        if load_in_8bit:
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, 
                                        load_in_8bit=load_in_8bit, 
                                        low_cpu_mem_usage=True,
                                        device_map="auto")
        elif load_in_4bit:
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, 
                                        load_in_4bit=load_in_4bit, 
                                        low_cpu_mem_usage=True,
                                        device_map="auto")
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, 
                                        low_cpu_mem_usage=True,
                                        device_map="auto")

        tokenizer = model_class.init_tokenizer(tokenizer_path)
        
        return model_class(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer
        )


    def forward(self, sources, max_length=512, beam_size=5):
        encoding = self.tokenizer(sources, return_tensors='pt').to(self.model.device)
        encoding['decoder_input_ids'] = encoding['input_ids'].clone()
        generated_ids = self.model.generate(**encoding,
                                            max_length=max_length, 
                                            num_beams=beam_size)

        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return predictions
    

    def predict(self, sources, max_length=512, beam_size=5):
        
        input_for_net = [' '.join(source.strip().split()).replace('\n', ' ') for source in sources]
        output = self.forward(input_for_net, max_length, beam_size)
       
        return output