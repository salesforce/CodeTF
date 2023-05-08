import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
import logging
from omegaconf import OmegaConf
from codetf.common.registry import registry
from codetf.models.base_model import BaseModel
from codetf.models.codet5_models.codet5_summarization import CodeT5Summarization
from codetf.models.codet5_models.codet5_nl2code import CodeT5NL2Code
from codetf.models.codet5_models.codet5_translation import CodeT5Translation
from codetf.models.codet5_models.codet5_refine import CodeT5Refine
from codetf.models.gpt_models.codegen_nl2code import CodeGenNL2Code


__all__ = [
    "CodeT5Summarization",
    "CodeT5NL2Code",
    "CodeT5Translation",
    "CodeT5Refine",
    "CodeGenNL2Code"
]

card_name_mapper = {
    "codet5_translation": "codet5",
    "codet5_summarization": "codet5",
    "codet5_nl2code": "codet5",
    "codet5_clone": "codet5",
    "codet5_defect": "codet5"
}

def get_model_config(model_name, model_type="base"):
    import json

    from omegaconf import OmegaConf

    config_path = registry.get_model_class(model_name).PRETRAINED_MODEL_CONFIG_DICT[
        model_type
    ]

    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config)

    # print(json.dumps(config, indent=4, sort_keys=True))

    return config


def construct_model_card(model_name, model_type=None, task=None, 
                        dataset=None, language=None,
                        quantize=None, quantize_algo=None):
    model_card_parts = [model_name]

    if model_type:
        model_card_parts.append(model_type)

    if dataset:
        model_card_parts.append(dataset)

    if task:
        model_card_parts.append(task)

    if language:
        model_card_parts.append(language)
    
    if quantize_algo:
        if quantize_algo is not "bitsandbyte":
            model_card_parts.append(quantize_algo)

    if quantize:
        model_card_parts.append(quantize)
    
    model_card_name = "-".join(model_card_parts)
    return model_card_name

def get_model_class_name(model_name, task):
    class_name = f"{model_name}_{task}"
    return class_name

def load_model(model_name, model_type="base", task="sum",
            dataset=None, language=None, is_eval=False, 
            quantize="int8", quantize_algo="bitsandbyte"):
    model_cls = registry.get_model_class(get_model_class_name(model_name,task))
    model_card = construct_model_card(model_name, model_type, task, dataset, language)
    model = model_cls.from_pretrained(model_card=model_card, quantize=quantize, quantize_algo=quantize_algo)
    # model = model_cls.load_model_from_config(config)
    if is_eval:
        model.eval()

    return model


class ModelZoo:
    def __init__(self) -> None:
        self.model_zoo = {
            k: list(v.PRETRAINED_MODEL_CONFIG_DICT.keys())
            for k, v in registry.mapping["model_name_mapping"].items()
        }

    def __str__(self) -> str:
        return (
            "=" * 50
            + "\n"
            + f"{'Architectures':<30} {'Types'}\n"
            + "=" * 50
            + "\n"
            + "\n".join(
                [
                    f"{name:<30} {', '.join(types)}"
                    for name, types in self.model_zoo.items()
                ]
            )
        )

    def __iter__(self):
        return iter(self.model_zoo.items())

    def __len__(self):
        return sum([len(v) for v in self.model_zoo.values()])


model_zoo = ModelZoo()
