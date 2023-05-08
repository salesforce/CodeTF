import warnings
from copy import deepcopy

import torch
import torch.nn.functional as F
from codetf.common.registry import registry
from codetf.models.codet5_models import CodeT5BaseModel
from torch import nn


@registry.register_model("codet5_sum")
class CodeT5Summarization(CodeT5BaseModel):
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "codet5-base-sum-python": "configs/inference/t5/codet5-base-sum-python.yaml",
        "codet5-base-sum-python-ctranslate-float16": "configs/inference/t5/codet5-base-sum-python-ctranslate-float16.yaml",
        "codet5-base-sum-python-ctranslate-int8": "configs/inference/t5/codet5-base-sum-python-ctranslate-int8.yaml",
        "codet5-base-sum-python-ctranslate-int8_float16": "configs/inference/t5/codet5-base-sum-python-ctranslate-int8float16",
        "codet5-base-sum-python-ctranslate-int16": "configs/inference/t5/codet5-base-sum-python-ctranslate-int16.yaml",
        "codet5-base-sum-go": "configs/inference/t5/codet5-base-sum-go.yaml",
        "codet5-base-sum-go-ctranslate-float16": "configs/inference/t5/codet5-base-sum-go-ctranslate-float16.yaml",
        "codet5-base-sum-go-ctranslate-int8": "configs/t5/inference/codet5-base-sum-go-ctranslate-int8.yaml",
        "codet5-base-sum-go-ctranslate-int8_float16": "configs/inference/t5/codet5-base-sum-go-ctranslate-int8float16",
        "codet5-base-sum-go-ctranslate-int16":  "configs/inference/t5/codet5-base-sum-go-ctranslate-int16.yaml",
        "codet5-base-sum-java": "configs/inference/t5/codet5-base-sum-java.yaml",
        "codet5-base-sum-java-ctranslate-float16": "configs/inference/t5/codet5-base-sum-java-ctranslate-float16.yaml",
        "codet5-base-sum-java-ctranslate-int8": "configs/inference/t5/codet5-base-sum-java-ctranslate-int8.yaml",
        "codet5-base-sum-java-ctranslate-int8_float16": "configs/inference/t5/codet5-base-sum-java-ctranslate-int8float16",
        "codet5-base-sum-java-ctranslate-int16": "configs/inference/t5/codet5-base-sum-java-ctranslate-int16.yaml",
        "codet5-base-sum-javascript": "configs/inference/t5/codet5-base-sum-javascript.yaml",
        "codet5-base-sum-javascript-ctranslate-float16": "configs/inference/t5/codet5-base-sum-javascript-ctranslate-float16.yaml",
        "codet5-base-sum-javascript-ctranslate-int8": "configs/inference/t5/codet5-base-sum-javascript-ctranslate-int8.yaml",
        "codet5-base-sum-javascript-ctranslate-int8_float16": "configs/inference/t5/codet5-base-sum-javascript-ctranslate-int8float16",
        "codet5-base-sum-javascript-ctranslate-int16": "configs/inference/t5/codet5-base-sum-javascript-ctranslate-int16.yaml",
        "codet5-base-sum-php": "configs/inference/t5/codet5-base-sum-php.yaml",
        "codet5-base-sum-php-ctranslate-float16": "configs/inference/t5/codet5-base-sum-php-ctranslate-float16.yaml",
        "codet5-base-sum-php-ctranslate-int8": "configs/inference/t5/codet5-base-sum-php-ctranslate-int8.yaml",
        "codet5-base-sum-php-ctranslate-int8_float16": "configs/inference/t5/codet5-base-sum-php-ctranslate-int8float16.yaml",
        "codet5-base-sum-php-ctranslate-int16": "configs/inference/t5/codet5-base-sum-php-ctranslate-int16.yaml",
        "codet5-base-sum-ruby": "configs/inference/t5/codet5-base-sum-ruby.yaml",
        "codet5-base-sum-ruby-ctranslate-float16": "configs/inference/t5/codet5-base-sum-ruby-ctranslate-float16.yaml",
        "codet5-base-sum-ruby-ctranslate-int8": "configs/inference/t5/codet5-base-sum-ruby-ctranslate-int8.yaml",
        "codet5-base-sum-ruby-ctranslate-int8_float16": "configs/inference/t5/codet5-base-sum-ruby-ctranslate-int8float16.yaml",
        "codet5-base-sum-ruby-ctranslate-int16": "configs/inference/t5/codet5-base-sum-ruby-ctranslate-int16.yaml"
    }

    def __init__(self, model, class_config, tokenizer):
        super().__init__()

        self.task = "summarization"
        self.codet5_model = model
        self.tokenizer = tokenizer
        self.max_source_length = class_config.get("max_source_length")
        self.max_target_length = class_config.get("max_target_length")
        self.beam_size = class_config.get("beam_size")

