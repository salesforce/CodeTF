

import torch
import torch.nn.functional as F
from datasets import load_dataset
from codetf.data_utility.base_dataset import BaseDataset, CustomDataset
from torch.utils.data import TensorDataset
import re


class HumanEvalDataset(BaseDataset):

    def __init__(self, tokenizer, max_length=448):
        super().__init__(tokenizer, max_length)
    
    def load(self):
        dataset = self.dataset_config["openai_humaneval"]

        # since humaneval only contains the "test" part
        dataset = load_dataset(dataset)["test"]

        prompts = []
        references = []
        num_tasks = len(dataset)
        for task_index in range(num_tasks):
            # without strip, the model generate commented codes ...
            if task_index < 5:
                prompts.append(self.tokenizer.eos_token + dataset[task_index]["prompt"].strip())

                unit_test = dataset[task_index]["test"]
                unit_test = re.sub(r'METADATA = {[^}]*}', '', unit_test, flags=re.MULTILINE)
                references.append(unit_test)
            # entry_point = f"check({dataset[task_index]['entry_point']})"
            # references.append("\n" + unit_test + "\n" + entry_point)
            # references.append("\n" + unit_test + "\n" + entry_point)

        prompt_token_ids, prompt_attention_masks = self.process_data(prompts, use_max_length=True, padding="max_length")
        # print(prompt_tensors)
        
        return TensorDataset(prompt_token_ids, prompt_attention_masks), references
    

    