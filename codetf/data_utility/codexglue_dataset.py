

import torch
import torch.nn.functional as F
from datasets import load_dataset
from codetf.data_utility.base_dataset import BaseDataset,CustomDataset
from torch.utils.data import TensorDataset

class CodeXGLUEDataset(BaseDataset):

    def __init__(self, tokenizer, max_length=512):
        
        super().__init__(tokenizer, max_length)
        self.load_funcs = {
            'text-to-code': self.load_codexglue_text_to_code_dataset,
            'code-to-text': self.load_codexglue_code_to_text_dataset,
            'java-to-csharp': self.load_codexglue_java_to_csharp_dataset,
            'code-refinement': self.load_codexglue_code_refinement_dataset
        }
    
    def load(self, subset):
        if subset in self.load_funcs:
            return self.load_funcs[subset]()
        else:
            raise ValueError(f'Invalid subset {subset}. Available subsets are: {list(self.load_funcs.keys())}')

    def load_codexglue_text_to_code_dataset(self):
        dataset = self.dataset_config["codexglue_text_to_code"]
        dataset = load_dataset(dataset)

        train = dataset["train"]
        train_nl_tensors, _ = self.process_data(train["nl"])
        train_code_tensors, _ = self.process_data(train["code"])
        
        test = dataset["test"]
        test_nl_tensors, _ = self.process_data(test["nl"])
        test_code_tensors, _ = self.process_data(test["code"])

        validation = dataset["validation"]
        validation_nl_tensors, _ = self.process_data(validation["nl"])
        validation_code_tensors, _ = self.process_data(validation["code"])

        return (train_nl_tensors, train_code_tensors), (test_nl_tensors, test_code_tensors), (validation_nl_tensors, validation_code_tensors)
    
    def load_codexglue_code_to_text_dataset(self):
        dataset = self.dataset_config["codexglue_code_to_text"]
        dataset = load_dataset(dataset)

        train = dataset["train"]
        train_code_tensors, _ = self.process_data(train["code"])
        train_docstring_tensors, _ = self.process_data(train["docstring"])
        
        test = dataset["test"]
        test_code_tensors, _ = self.process_data(test["code"])
        test_docstring_tensors, _ = self.process_data(test["docstring"])

        validation = dataset["validation"]
        validation_code_tensors, _ = self.process_data(validation["code"])
        validation_docstring_tensors, _ = self.process_data(validation["docstring"])

        return (train_code_tensors, train_docstring_tensors), (test_code_tensors, test_docstring_tensors), (validation_code_tensors, validation_docstring_tensors)

    def load_codexglue_java_to_csharp_dataset(self):
        dataset = self.dataset_config["codexglue_java_to_csharp"]
        dataset = load_dataset(dataset)

        train = dataset["train"]
        train_java_tensors, _ = self.process_data(train["java"])
        train_csharp_tensors, _ = self.process_data(train["cs"])
        
        test = dataset["test"]
        test_java_tensors, _ = self.process_data(test["java"])
        test_csharp_tensors, _ = self.process_data(test["cs"])

        validation = dataset["validation"]
        validation_java_tensors, _ = self.process_data(validation["java"])
        validation_csharp_tensors, _ = self.process_data(validation["cs"])

        return (train_java_tensors, train_csharp_tensors), (test_java_tensors, test_csharp_tensors), (validation_java_tensors, validation_csharp_tensors)

    def load_codexglue_code_refinement_dataset(self):
        dataset = self.dataset_config["codexglue_code_refinement"]
        dataset = load_dataset(dataset)

        train = dataset["train"]
        train_buggy_tensors, _ = self.process_data(train["buggy"])
        train_fixed_tensors, _ = self.process_data(train["fixed"])
        
        test = dataset["test"]
        test_buggy_tensors, _ = self.process_data(test["buggy"])
        test_fixed_tensors, _ = self.process_data(test["fixed"])

        validation = dataset["validation"]
        validation_buggy_tensors, _ = self.process_data(validation["buggy"])
        validation_fixed_tensors, _ = self.process_data(validation["fixed"])

        return (train_buggy_tensors, train_fixed_tensors), (test_buggy_tensors, test_fixed_tensors), (validation_buggy_tensors, validation_fixed_tensors)
