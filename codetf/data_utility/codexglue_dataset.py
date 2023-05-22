

import torch
import torch.nn.functional as F
from datasets import load_dataset
from codetf.data_utility.base_dataloader import BaseDataLoader,CustomDataset
from torch.utils.data import TensorDataset

class CodeXGLUEDataset(BaseDataLoader):

    def __init__(self, tokenizer, max_length=256):
        
        super().__init__(tokenizer,max_length)
    
    def load_codexglue_text_to_code_dataset(self):
        dataset = self.dataset_config["codexglue_text_to_code"]
        dataset = load_dataset(dataset)

        train = dataset["validation"]
        train_nl_tensors = self.process_data(train["nl"])
        train_code_tensors = self.process_data(train["code"])
        
        test = dataset["test"]
        test_nl_tensors = self.process_data(test["nl"])
        test_code_tensors = self.process_data(test["code"])

        validation = dataset["validation"]
        validation_nl_tensors = self.process_data(validation["nl"])
        validation_code_tensors = self.process_data(validation["code"])

        # train_dataset = TensorDataset(train_nl_tensors, train_code_tensors)
        # test_dataset =  TensorDataset(test_nl_tensors, test_code_tensors)
        # validation_dataset= TensorDataset(validation_nl_tensors, validation_code_tensors)

        train_dataset = CustomDataset(train_nl_tensors, train_code_tensors)
        test_dataset =  CustomDataset(test_nl_tensors, test_code_tensors)
        validation_dataset= CustomDataset(validation_nl_tensors, validation_code_tensors)


        return train_dataset, test_dataset, validation_dataset
    
    def load_codexglue_code_to_text_dataset(self):
        dataset = self.dataset_config["codexglue_code_to_text"]
        dataset = load_dataset(dataset)

        train = dataset["train"]
        train_code_tensors = self.process_data(train["code"])
        train_docstring_tensors = self.process_data(train["docstring"])
        
        test = dataset["test"]
        test_code_tensors = self.process_data(test["code"])
        test_docstring_tensors = self.process_data(test["docstring"])

        validation = dataset["validation"]
        validation_code_tensors = self.process_data(validation["code"])
        validation_docstring_tensors = self.process_data(validation["docstring"])

        train_dataset = CustomDataset(train_code_tensors, train_docstring_tensors)
        test_dataset =  CustomDataset(test_code_tensors, test_docstring_tensors)
        validation_dataset= CustomDataset(validation_code_tensors, validation_docstring_tensors)

        return train_dataset, test_dataset, validation_dataset
    
    def load_codexglue_java_to_csharp_dataset(self):
        dataset = self.dataset_config["codexglue_java_to_csharp"]
        dataset = load_dataset(dataset)

        train = dataset["validation"]
        train_java_tensors = self.process_data(train["java"])
        train_csharp_tensors = self.process_data(train["cs"])
        
        test = dataset["test"]
        test_java_tensors = self.process_data(test["java"])
        test_csharp_tensors = self.process_data(test["cs"])

        validation = dataset["validation"]
        validation_java_tensors = self.process_data(validation["java"])
        validation_csharp_tensors = self.process_data(validation["cs"])

        train_dataset =  CustomDataset(train_java_tensors, train_csharp_tensors)
        test_dataset =  CustomDataset(test_java_tensors, test_csharp_tensors)
        validation_dataset= CustomDataset(validation_java_tensors, test_csharp_tensors)

        return train_dataset, test_dataset, validation_dataset
    
    def load_codexglue_code_refinement_dataset(self):
        dataset = self.dataset_config["codexglue_code_refinement"]
        dataset = load_dataset(dataset)

        train = dataset["validation"]
        train_buggy_tensors = self.process_data(train["buggy"])
        train_fixed_tensors = self.process_data(train["fixed"])
        
        test = dataset["test"]
        test_buggy_tensors = self.process_data(test["buggy"])
        test_fixed_tensors = self.process_data(test["fixed"])

        validation = dataset["validation"]
        validation_buggy_tensors = self.process_data(validation["buggy"])
        validation_fixed_tensors = self.process_data(validation["fixed"])

        train_dataset =  CustomDataset(train_buggy_tensors, train_fixed_tensors)
        test_dataset =  CustomDataset(test_buggy_tensors, test_fixed_tensors)
        validation_dataset= CustomDataset(validation_buggy_tensors, validation_fixed_tensors)

        return train_dataset, test_dataset, validation_dataset



