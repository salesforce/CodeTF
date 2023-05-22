import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
import torch
from codetf.trainer.causal_lm_trainer import CausalLMTrainer
from codetf.data_utility.codexglue_dataset import CodeXGLUEDataset
from codetf.models import load_model_pipeline
from codetf.performance.evaluation_metric import EvaluationMetric
from codetf.data_utility.base_dataset import CustomDataset

model_class = load_model_pipeline(model_name="causal-lm", task="pretrained",
            model_type="codegen-350M-mono", is_eval=False,
            load_in_8bit=False, weight_sharding=False)


dataset = CodeXGLUEDataset(tokenizer=model_class.get_tokenizer())
train, test, validation = dataset.load(subset="text-to-code")

train_dataset = CustomDataset(train[0], train[1])
test_dataset= CustomDataset(test[0], test[1])
val_dataset= CustomDataset(validation[0], validation[1])

evaluator = EvaluationMetric(metric="bleu", tokenizer=model_class.tokenizer)

# peft can be in ["lora", "prefixtuning"]
trainer = CausalLMTrainer(train_dataset=train_dataset, 
                        validation_dataset=val_dataset, 
                        peft=None,
                        pretrained_model_or_path=model_class.get_model(),
                        tokenizer=model_class.get_tokenizer())
trainer.train()
# trainer.evaluate(test_dataset=test_dataset)





