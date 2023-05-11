import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
import torch
from codetf.trainer.codet5_trainer import CodeT5Seq2SeqTrainer
from codetf.data_utility.codexglue_dataloader import CodeXGLUEDataLoader
from codetf.models import load_model

model_class = load_model(model_name="codet5", 
                model_type="base", task="sum", language="python", 
                is_eval=True, quantize="int8", quantize_algo="bitsandbyte")


dataloader = CodeXGLUEDataLoader(tokenizer=model_class.tokenizer)
train_dataset, test_dataset, val_dataset = dataloader.load_codexglue_text_to_code_dataset()

# peft can be in ["lora", "prefixtuning"]
trainer = CodeT5Seq2SeqTrainer(train_dataset=train_dataset, 
                                validation_dataset=val_dataset, 
                                peft="prefixtuning",
                                pretrained_model_or_path=model_class.model,
                                tokenizer=model_class.tokenizer)
trainer.train()
trainer.evaluate(test_dataset=test_dataset)


