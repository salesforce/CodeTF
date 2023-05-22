import multiprocessing
from torch.utils.data import TensorDataset
import torch
import concurrent.futures
from omegaconf import OmegaConf
from codetf.common.utils import get_abs_path
from torch.utils.data import IterableDataset, Dataset

class BaseDataset():
    
    DATASET_CONFIG_PATH = "configs/dataset/dataset.yaml"

    def __init__(self, tokenizer, max_length=512):
        
        self.max_length = max_length
        self.tokenizer = tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.dataset_config = self.load_dataset_config_dict()

    def load_dataset_config_dict(self):
        dataset_config = OmegaConf.load(get_abs_path(self.DATASET_CONFIG_PATH)).dataset
        return dataset_config

    def process_data(self, data, padding="max_length", truncation=True):
        outputs = self.tokenizer(data, padding=padding, truncation=truncation, return_tensors="pt", max_length=self.max_length)
        return outputs["input_ids"], outputs["attention_mask"]
    
class CustomDataset(Dataset):

    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

