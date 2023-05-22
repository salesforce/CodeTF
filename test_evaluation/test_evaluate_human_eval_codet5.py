import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from codetf.models import load_model_pipeline
from codetf.data_utility.util import EOF_STRINGS, EndOfFunctionCriteria, remove_last_block
from torch.utils.data.dataloader import DataLoader
from transformers import StoppingCriteriaList
from codetf.data_utility.human_eval_dataset import HumanEvalDataset
import torch
import os
from accelerate import Accelerator
import torch
from collections import defaultdict
from tqdm import tqdm
import torch
from evaluate import load
from codetf.performance.model_evaluator import ModelEvaluator
from torch.utils.data import TensorDataset

def main():
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_class = load_model_pipeline(model_name="codet5", task="nl2code",
                model_type="base", is_eval=True,
                load_in_8bit=True, weight_sharding=False)

    dataset = HumanEvalDataset(tokenizer=model_class.get_tokenizer())
    prompt_token_ids, prompt_attention_masks, references= dataset.load()

    problems = TensorDataset(prompt_token_ids, prompt_attention_masks)
    
    evaluator = ModelEvaluator(model_class)
    avg_pass_at_k = evaluator.evaluate_pass_k(problems=problems, unit_tests=references)
    print("Pass@k: ", avg_pass_at_k)

if __name__ == "__main__":
    main()