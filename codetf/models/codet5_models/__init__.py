import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from transformers import RobertaTokenizer
from codetf.models.base_model import BaseModel
from transformers import T5ForConditionalGeneration
from codetf.common.registry import registry

@registry.register_model("codet5")
class CodeT5Seq2SeqModel(BaseModel):
    
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
        return RobertaTokenizer.from_pretrained(model)
    
  
    @classmethod
    def load_model_from_config(model_class, model_config, quantize=None, quantize_algo=None):
        
        # model = ctranslate2.Translator(huggingface_path, compute_type="int8")
        if quantize_algo:
            if quantize_algo == "bitsandbyte":
                if quantize == "int8":
                    print("Loading 8 bit....")
                    model = T5ForConditionalGeneration.from_pretrained(model_config["huggingface_url"], load_in_8bit=True, device_map='auto')
                else:
                    model = T5ForConditionalGeneration.from_pretrained(model_config.get("huggingface_url"), device_map='auto')
            else:
                model = ctranslate2.Translator(model_config["huggingface_url"], compute_type=quantize)
        else:
            model = T5ForConditionalGeneration.from_pretrained(model_config["huggingface_url"], device_map='auto')
        
           
        tokenizer = model_class.init_tokenizer(model_config.get("tokenizer_url"), device_map='auto')

        return model_class(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer
        )
    

    def forward_seq2seq(self, sources):
        input_ids = self.tokenizer(sources, padding=True, return_tensors='pt').input_ids.to(self.device)
        generated_ids = self.model.generate(input_ids, 
                                            max_length=self.max_prediction_length, 
                                            num_beams=self.beam_size, 
                                            num_return_sequences=1)

        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return predictions
    

    def predict(self, sources):
        
        input_for_net = [' '.join(source.strip().split()).replace('\n', ' ') for source in sources]
        # if self.task in ["sum", "translate", "nl2code", "refine"]:
        output = self.forward_seq2seq(input_for_net)
       
        return output