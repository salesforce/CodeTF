

import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset
from accelerate import Accelerator,DistributedType
from torch.optim.lr_scheduler import OneCycleLR
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments,logging,set_seed, get_linear_schedule_with_warmup, AdamW
from omegaconf import OmegaConf
# from accelerate import Accelerator
from codetf.trainer.base_trainer import BaseTrainer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, AdaLoraConfig, prepare_model_for_int8_training
from codetf.common.utils import get_abs_path
from transformers.trainer_pt_utils import get_parameter_names
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sacrebleu
import os 

class CodeT5Seq2SeqTrainer(BaseTrainer):    
    def __init__(self, train_dataset, validation_dataset=None, tokenizer=None, 
                checkpoints_path="./checkpoints", pretrained_model_or_path="Salesforce/codet5-base-multi-sum", 
                training_args=None, evaluator=None, evaluation_fn=None, peft=None):
        
        # model = T5ForConditionalGeneration.from_pretrained(pretrained_model_or_path)
      
            
        super().__init__(pretrained_model_or_path, tokenizer, train_dataset, validation_dataset,
                        checkpoints_path, pretrained_model_or_path,
                        evaluator, evaluation_fn)
        
        if training_args is None:
            self.training_args = self.get_default_codet5_hyperparameters()
        else:
            self.training_args = training_args

        self.trainer = self.init_trainer()

        if peft:
            self.model = prepare_model_for_int8_training(self.model)
            if peft == "lora":
                peft_config = self.get_default_lora_config_for_codet5()
            self.model.enable_input_require_grads()
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
    
