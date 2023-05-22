import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
import torch
from codetf.trainer.causal_lm_trainer import CausalLMTrainer
from codetf.data_utility.codexglue_dataset import CodeXGLUEDataset
from codetf.models import load_model_pipeline
from codetf.performance.evaluation_metric import EvaluationMetric
from codetf.data_utility.base_dataset import CustomDataset
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=True)
dataset = CodeXGLUEDataset(tokenizer=tokenizer)
train, test, validation = dataset.load(subset="text-to-code")

train_dataset= CustomDataset(train[0], train[1])
test_dataset= CustomDataset(test[0], test[1])
val_dataset= CustomDataset(validation[0], validation[1])

