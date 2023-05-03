import warnings
from copy import deepcopy

import torch
import torch.nn.functional as F
from codetf.common.registry import registry
from codetf.models.codet5_models import CodeT5BaseModel
from torch import nn
from transformers import T5ForConditionalGeneration

@registry.register_model("codet5_translation")
class CodeT5Translation(CodeT5BaseModel):
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "codet5-base-translate-cs-java": "configs/t5/codet5-base-translate-cs-java.yaml",
        "codet5-base-translate-java-cs": "configs/t5/java-cs.yaml"
    }

    def __init__(self, model, class_config, tokenizer):
        super().__init__()

        self.task = "translation"
        self.codet5_model = model
        self.tokenizer = tokenizer
        self.max_source_length = class_config.get("max_source_length")
        self.max_target_length = class_config.get("max_target_length")
        self.beam_size = class_config.get("beam_size")