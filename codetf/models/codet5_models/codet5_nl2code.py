import warnings
from copy import deepcopy

import torch
import torch.nn.functional as F
from codetf.common.registry import registry
from codetf.models.codet5_models import CodeT5BaseModel
from torch import nn
from transformers import T5ForConditionalGeneration

@registry.register_model("codet5_nl2code")
class CodeT5NL2Code(CodeT5BaseModel):
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/t5/codet5_nl2code_base.yaml"
    }

    def __init__(self, model, class_config, tokenizer):
        super().__init__()

        self.task = "nl2code"
        self.codet5_model = model
        self.tokenizer = tokenizer
        self.max_source_length = class_config.get("max_source_length")
        self.max_target_length = class_config.get("max_target_length")
        self.beam_size = class_config.get("beam_size")


    def forward(self, sources):
        input_ids = self.tokenizer(sources, padding=True, return_tensors='pt').input_ids.to(self.device)
        generated_ids = self.codet5_model.generate(input_ids, 
                                            max_length=self.max_target_length, 
                                            num_beams=self.beam_size, 
                                            num_return_sequences=1)

        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return predictions

    def predict(self, sources):
        input_for_net = [' '.join(source.strip().split()).replace('\n', ' ') for source in sources]
        output = self.forward(input_for_net)
        return output
   