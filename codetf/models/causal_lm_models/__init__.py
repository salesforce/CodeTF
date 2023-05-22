import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from codetf.models.base_model import BaseModel
from codetf.common.registry import registry
from accelerate import Accelerator
from collections import defaultdict
from tqdm import tqdm
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import hf_hub_download
import torch

@registry.register_model("causal-lm")
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
    def load_model_from_config(model_class, model_config, load_in_8bit=True, weight_sharding=True):
        checkpoint = model_config["huggingface_url"]
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
                                            device_map={"": Accelerator().process_index})
            else:
                model = AutoModelForCausalLM.from_pretrained(checkpoint, 
                                            device_map={"": Accelerator().process_index})


        tokenizer = model_class.init_tokenizer(checkpoint)
        
        return model_class(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer
        )
   
    def forward(self, sources):
        input_ids = self.tokenizer(sources, return_tensors='pt').input_ids.to(self.device)
        generated_ids = self.model.generate(input_ids, 
                                            max_length=self.max_prediction_length)

        predictions = self.tokenizer.batch_decode(generated_ids, truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
        return predictions

    def predict(self, sources):
        input_for_net = [' '.join(source.strip().split()).replace('\n', ' ') for source in sources]
        output = self.forward(input_for_net)
        return output
    
    # def predict_for_code_complete(self, dataloader, args, batch_size=1):
    #     gen_token_dict = defaultdict(list)  # dict of list of generated tokens
    #     for step, batch in tqdm(enumerate(dataloader)):
    #         # print(batch)
    #         prompt_ids, attention_masks = batch
           
    #         with torch.no_grad():
    #             args["stopping_criteria"][0].start_length = attention_masks[0].sum().item()
    #             # print("DDD : ",attention_masks[0].sum().item())
    #             # print(attention_masks.sum().item())
    #             input_ids = prompt_ids[0, :attention_masks[0].sum().item()]
    #             print("-----------")
    #             print(input_ids)
    #             input_data = self.tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #             print("Input: ", input_data)

    #             # print(input_ids.shape)
    #             generated_tokens = self.model.generate(
    #                 input_ids=input_ids.unsqueeze(0), max_length=512, num_return_sequences=batch_size, **args
    #             )
    #             generated_tokens = generated_tokens.cpu().numpy()
    #             # print(generated_tokens)
    #             gen_code = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #             print("Solution: ", gen_code)
    #             # for task, generated_tokens in zip(generated_tasks, generated_tokens):
    #             #     gen_token_dict[task].append(generated_tokens)
    #     # code_gens = [[] for _ in range(n_tasks)]
    #     # for task, generated_tokens in gen_token_dict.items():
    #     #     for s in generated_tokens:
    #     #         gen_code = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #     #         code_gens[task].append(remove_last_block(gen_code))
    #     return code_gens
