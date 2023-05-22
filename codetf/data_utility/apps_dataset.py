

import torch
import torch.nn.functional as F
from datasets import load_dataset
from codetf.data_utility.base_dataloader import BaseDataset,CustomDataset
from torch.utils.data import TensorDataset

class APPSDataset(BaseDataset):

    def __init__(self, tokenizer, max_length=256):
        
        super().__init__(tokenizer,max_length)
    
    def load_dataset(self):
        dataset = self.dataset_config["APPS"]
        dataset = load_dataset(dataset)

        train = dataset["train"]
        train_question_tensors = self.process_data(train["question"])
        train_solutions_tensors = self.process_data(train["solution"])
        train_input_output = train["input_output"]

        test = dataset["test"]
        test_question_tensors = self.process_data(train["question"])
        test_solutions_tensors = self.process_data(train["solution"])
        test_input_output = train["input_output"]

        train_dataset = TensorDataset(train_question_tensors, train_solutions_tensors)
        test_dataset =  TensorDataset(test_question_tensors, test_solutions_tensors)
        validation_dataset= TensorDataset(validation_nl_tensors, validation_code_tensors)

        return train_dataset, test_dataset, validation_dataset
    

    