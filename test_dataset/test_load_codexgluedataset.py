import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
import torch
from codetf.data_utility.codexglue_dataset import CodeXGLUEDataset
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=True)
dataset = CodeXGLUEDataset(tokenizer=tokenizer)
train, test, validation = dataset.load(subset="text-to-code")

