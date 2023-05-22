import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from transformers import RobertaTokenizer
from codetf.models.base_model import BaseModel
from transformers import T5ForConditionalGeneration, T5Config
from codetf.common.registry import registry
from accelerate import Accelerator

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
    def load_model_from_config(model_class, model_config, load_in_8bit=True, weight_sharding=True):
        
        checkpoint = model_config["huggingface_url"]
        if weight_sharding:
            weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")
            config = T5Config.from_pretrained(checkpoint)
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config)

            model.tie_weights()            
            model = load_checkpoint_and_dispatch(
                model, weights_location, device_map="auto", no_split_module_classes=["GPTJBlock"]
            )
        else:
            if load_in_8bit:
                model = T5ForConditionalGeneration.from_pretrained(checkpoint, 
                                            load_in_8bit=load_in_8bit, 
                                            device_map="auto")
            else:
                model = T5ForConditionalGeneration.from_pretrained(checkpoint, 
                                            device_map="auto")

           
        tokenizer = model_class.init_tokenizer(model_config.get("tokenizer_url"))

        return model_class(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer
        )
    

    def forward_seq2seq(self, sources):
        encoding = self.tokenizer(sources, return_tensors='pt')
        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)
        generated_ids = self.model.generate(input_ids, attention_mask=attention_mask,
                                            max_length=self.max_prediction_length, 
                                            num_beams=self.beam_size)

        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return predictions
    

    def predict(self, sources):
        
        input_for_net = [' '.join(source.strip().split()).replace('\n', ' ') for source in sources]
        # if self.task in ["sum", "translate", "nl2code", "refine"]:
        output = self.forward_seq2seq(input_for_net)
       
        return output