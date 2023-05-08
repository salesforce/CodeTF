import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
from codetf.trainer.codet5_trainer import CodeT5Seq2SeqTrainer
from codetf.data_utility.codexglue_dataloader import CodeXGLUEDataLoader
from transformers import RobertaTokenizer


tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
dataloader = CodeXGLUEDataLoader(tokenizer=tokenizer)
train_dataset, test_dataset, val_dataset = dataloader.load_codexglue_text_to_code_dataset()


# peft can be in ["lora", "prefixtuning"]
trainer = CodeT5Seq2SeqTrainer(train_dataset=train_dataset, validation_dataset=val_dataset, tokenizer=tokenizer, peft="prefixtuning")
trainer.train()