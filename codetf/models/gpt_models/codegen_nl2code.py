import warnings
from copy import deepcopy

import torch
import torch.nn.functional as F
from codetf.common.registry import registry
from codetf.models.gpt_models import GPTBaseModel
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

@registry.register_model("codegen_nl2code")
class CodeGenNL2Code(GPTBaseModel):
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "350M-nl": "configs/gpt/codegen_nl2code_350M-nl.yaml",
        "2B-mono": "configs/gpt/codegen_nl2code_2B-mono.yaml",
    }

    def __init__(self, model, max_source_length, max_target_length, beam_size, tokenizer_path):
        super().__init__()

        self.codegen_model = model
        self.tokenizer = self.init_tokenizer(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.beam_size = beam_size


    def forward(self, sources):
        input_ids = self.tokenizer(sources, padding=True, return_tensors='pt').to(self.device)
        generated_ids = self.codegen_model.generate(**input_ids, 
                                            max_length=self.max_target_length)

        predictions = self.tokenizer.batch_decode(generated_ids, truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
        return predictions

    def predict(self, sources):
        input_for_net = [' '.join(source.strip().split()).replace('\n', ' ') for source in sources]
        output = self.forward(input_for_net)
        return output
    
   