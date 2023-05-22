

import torch
import torch.nn.functional as F
from datasets import load_dataset
from codetf.data_utility.base_dataloader import BaseDataset
# from torch.utils.data import TensorDataset

class MBPPDataset(BaseDataset):

    def __init__(self, tokenizer, max_length=512):
        
        super().__init__(tokenizer, max_length)
    
    def load(self):
        dataset = self.dataset_config["mbpp"]
        dataset = load_dataset(dataset)

        train = dataset["train"]
        train_question_tensors = self.process_data(train["question"])
        train_solutions_tensors = self.process_data(train["solution"])
        train_input_output = train["input_output"]

        test = dataset["test"]
        test_question_tensors = self.process_data(test["question"])
        test_solutions_tensors = self.process_data(test["solution"])
        test_input_output = test["input_output"]

        train_data = (train_question_tensors, train_solutions_tensors, train_input_output)
        test_data = (test_question_tensors, test_solutions_tensors, test_input_output)

        return train_data, test_data
        

    