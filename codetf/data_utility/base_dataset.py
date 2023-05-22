import multiprocessing
from torch.utils.data import TensorDataset
import torch
import concurrent.futures
from omegaconf import OmegaConf
from codetf.common.utils import get_abs_path
from torch.utils.data import IterableDataset, Dataset

class BaseDataset():
    
    DATASET_CONFIG_PATH = "configs/dataset/dataset.yaml"

    def __init__(self, tokenizer, max_length=256):
        
        self.max_length = max_length
        self.tokenizer = tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.dataset_config = self.load_dataset_config_dict()

    def load_dataset_config_dict(self):
        dataset_config = OmegaConf.load(get_abs_path(self.DATASET_CONFIG_PATH)).dataset
        return dataset_config

    def process_data(self, data, use_max_length=False, padding=False, truncation=False):
        
        data_tuples = []
        for d in data:
            data_tuple = (d, use_max_length, padding, truncation)
            data_tuples.append(data_tuple)

        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()-5) as executor:
            features = list(executor.map(lambda f: self._process_text(*f), tuple(data_tuples)))	

        # for f in features:
        #     print(f["attention_mask"])
            # print(f["attention_mask"].shape)
            # print(f["input_ids"].shape)

        token_ids = torch.stack([f["input_ids"] for f in features]).squeeze(1)
        attention_masks = torch.stack([f["attention_mask"][0] for f in features]).squeeze(1)
        return token_ids,attention_masks
    
    def _process_text(self, text, use_max_length=False, padding=False, truncation=False):  
        if use_max_length:
            outputs = self.tokenizer(text, max_length=self.max_length, padding=padding, truncation=truncation, return_tensors="pt")
        else:
            outputs = self.tokenizer(text, padding=padding, truncation=truncation, return_tensors="pt")

        return outputs

class CustomDataset(Dataset):

    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

