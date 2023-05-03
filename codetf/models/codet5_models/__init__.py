import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from transformers import RobertaTokenizer
from codetf.models.base_model import BaseModel
from transformers import T5ForConditionalGeneration
import ctranslate2

class CodeT5BaseModel(BaseModel):

    @classmethod
    def init_tokenizer(cls, model):
        return RobertaTokenizer.from_pretrained(model)
    
  
    @classmethod
    def load_model_from_config(model_class, class_config, quantize=None, quantize_algo=None):
        
        # model = ctranslate2.Translator(huggingface_path, compute_type="int8")
        if quantize_algo:
            if quantize_algo == "bitsandbyte":
                if quantize == "int8":
                    print("Loading 8 bit....")
                    model = T5ForConditionalGeneration.from_pretrained(class_config.get("huggingface_url"), load_in_8bit=True, device_map='auto')
                else:
                    model = T5ForConditionalGeneration.from_pretrained(class_config.get("huggingface_url"))
            else:
                model = ctranslate2.Translator(class_config.get("huggingface_url"), compute_type=quantize)
        else:
            model = T5ForConditionalGeneration.from_pretrained(class_config.get("huggingface_url"))
           
        tokenizer = model_class.init_tokenizer(class_config.get("tokenizer_url"))

        return model_class(
            model=model,
            class_config=class_config,
            tokenizer=tokenizer
        )
    
    def forward_seq2seq(self, sources):
        input_ids = self.tokenizer(sources, padding=True, return_tensors='pt').input_ids.to(self.device)
        generated_ids = self.codet5_model.generate(input_ids, 
                                            max_length=self.max_target_length, 
                                            num_beams=self.beam_size, 
                                            num_return_sequences=1)

        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return predictions
    

    def predict(self, sources):
        
        input_for_net = [' '.join(source.strip().split()).replace('\n', ' ') for source in sources]
        if self.task in ["summarization", "translation", "nl2code"]:
            output = self.forward_seq2seq(input_for_net)
       
        return output