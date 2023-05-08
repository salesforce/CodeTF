from omegaconf import OmegaConf

import numpy as np
import os
import torch
import torch.nn as nn
import requests
from tqdm import tqdm
from urllib.parse import urlsplit
from codetf.common.utils import get_abs_path
import urllib.request

def download_model(model_cache_path, model_url):
    if not os.path.exists(model_cache_path):
        with urllib.request.urlopen(model_url) as response, open(model_cache_path, 'wb') as out_file:
            total_size = int(response.getheader('Content-Length'))
            chunk_size = 1024
            progress = tqdm(total=total_size, unit='B', unit_scale=True, desc=model_cache_path)
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                progress.update(len(chunk))
            progress.close()

class BaseModel(nn.Module):
    """Base class for models."""


    DEFAULT_CONFIG_PATH = "configs/default.yaml"
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[0].device

    @classmethod
    def from_pretrained(model_class, model_card, quantize="int8", quantize_algo="bitsandbyte"):
        """
        Build a pretrained model from default configuration file, specified by model_type.
        """
        model_config = OmegaConf.load(model_class.get_class_config_path(model_card)).model
        default_config = OmegaConf.load(get_abs_path(model_class.DEFAULT_CONFIG_PATH)).env

        model_cls = model_class.load_model_from_config(class_config=model_config, quantize=quantize, quantize_algo=quantize_algo)

        return model_cls


    @classmethod
    def get_class_config_path(cls, model_card):
        # print(cls.PRETRAINED_MODEL_CONFIG_DICT)
        assert (
            model_card in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_card)
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_card])

    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot
