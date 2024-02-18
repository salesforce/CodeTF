import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
import logging
from omegaconf import OmegaConf
from codetf.common.registry import registry
from codetf.models.base_model import BaseModel
from codetf.models.seq2seq_models import Seq2SeqModel
from codetf.models.causal_lm_models import CausalLMModel
from codetf.models.bert_models import BertModel
from codetf.common.utils import get_abs_path

__all__ = [
    "Seq2SeqModel",
    "CausalLMModel",
    "BertModel"
]

def construct_model_card(model_name, model_type=None, task=None, 
                        dataset=None, language=None):
    model_card_parts = [model_name]

    if model_type:
        model_card_parts.append(model_type)

    if dataset:
        model_card_parts.append(dataset)

    if task:
        model_card_parts.append(task)

    if language:
        model_card_parts.append(language)
    
    model_card_name = "-".join(model_card_parts)
    return model_card_name

def get_model_class_name(model_name, task):
    class_name = f"{model_name}_{task}"
    return class_name

def load_model_pipeline(model_name, model_type="base", task="sum",
            dataset=None, language=None, is_eval=True, 
            load_in_8bit=False, load_in_4bit=False, weight_sharding=False):
    
    model_cls = registry.get_model_class(model_name)
    model_card = construct_model_card(model_name, model_type, task, dataset, language)
    model = model_cls.from_pretrained(model_card=model_card, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, weight_sharding=weight_sharding)
    if is_eval:
        model.eval()

    return model

def load_model_from_path(checkpoint_path, tokenizer_path, model_name, is_eval=True, load_in_8bit=False, load_in_4bit=False):
    model_cls = registry.get_model_class(model_name)
    model = model_cls.from_custom(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
    if is_eval:
        model.eval()

    return model

class ModelZoo:
    def __init__(self, config_files):
        self.config_files = config_files
        self.models = self.load_models()

    def load_models(self):
        models = {}

        for file in self.config_files:
            try:
                data = OmegaConf.load(get_abs_path(file))
                for model in data:
                    model_name, model_type, task = self.parse_model_key(model)
                    if model_name not in models:
                        models[model_name] = []
                    models[model_name].append((model_type, task))
            except Exception as exc:
                print(exc)

        return models

    @staticmethod
    def parse_model_key(key):
        parts = key.split("-")
        model_name = parts[0]
        model_type = "-".join(parts[1:-1])
        task = parts[-1]
        return model_name, model_type, task

    def __str__(self):
        output = "============================================================================================================\n"
        output += "Architectures                  Types                           Tasks\n"
        output += "============================================================================================================\n"
        for model_name, model_infos in self.models.items():
            for i, model_info in enumerate(model_infos):
                model_type, task = model_info
                if i == 0:
                    output += f"{model_name:<30} {model_type:<30} {task:<30}\n"
                else:
                    output += f"{'':<30} {model_type:<30} {task:<30}\n"
        return output


# instantiate the ModelZoo at the module level
model_zoo = ModelZoo(["configs/inference/causal_lm.yaml", 
                      "configs/inference/codet5.yaml", 
                      "configs/inference/bert.yaml"])
