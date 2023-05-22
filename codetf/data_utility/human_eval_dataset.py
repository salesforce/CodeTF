import torch
import torch.nn.functional as F
from datasets import load_dataset
from codetf.data_utility.base_dataset import BaseDataset
import re

class HumanEvalDataset(BaseDataset):

    def __init__(self, tokenizer, max_length=512):
        super().__init__(tokenizer, max_length)
    
    def load(self):
        dataset = self.dataset_config["openai_humaneval"]

        # since humaneval only contains the "test" part
        dataset = load_dataset(dataset)["test"]

        prompts = []
        references = []
        num_tasks = len(dataset)
        for task_index in range(num_tasks):
            # without strip, the model generates commented codes ...
            prompts.append(self.tokenizer.eos_token + dataset[task_index]["prompt"].strip())

            unit_test = dataset[task_index]["test"]
            unit_test = re.sub(r'METADATA = {[^}]*}', '', unit_test, flags=re.MULTILINE)
            references.append(unit_test)

        prompt_token_ids, prompt_attention_masks = self.process_data(prompts, use_max_length=True, padding="max_length")
        
        return prompt_token_ids, prompt_attention_masks, references
    

    