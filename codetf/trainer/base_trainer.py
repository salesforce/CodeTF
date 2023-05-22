
import torch
import torch.nn.functional as F
import torch.nn as nn
from datasets import load_dataset
from accelerate import Accelerator,DistributedType
from torch.optim.lr_scheduler import OneCycleLR
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments,logging,set_seed
from omegaconf import OmegaConf
from accelerate import Accelerator
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, AdaLoraConfig, PrefixTuningConfig, PromptTuningInit, PromptTuningConfig
from codetf.common.utils import get_abs_path
import sacrebleu
from transformers.trainer_pt_utils import get_parameter_names
import os 

class BaseTrainer():
    
    DEFAULT_CODET5_HYPERPARAMETERS = "configs/training/codet5.yaml"
    DEFAULT_CAUSAL_LM_HYPERPARAMETERS = "configs/training/causal_lm.yaml"

    def __init__(self, model, tokenizer, train_dataset, validation_dataset=None,
                checkpoints_path="./checkpoints", pretrained_model_or_path=None,
                evaluator=None, evaluation_fn=None):
        
        self.checkpoints_path = checkpoints_path
        self.create_checkpoints_path(checkpoints_path)
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        # check for evaluator and evaluation_fn, cannot co-exist
        if evaluator is not None and evaluation_fn is not None:
            raise ValueError("evaluator and evaluation_fn cannot co-exist. Please choose one.")

        if evaluator is not None:
            self.compute_metrics_fn = evaluator.compute
        elif evaluation_fn is not None:
            self.compute_metrics_fn = evaluation_fn
        else:
            self.compute_metrics_fn = None

    def init_trainer(self):
        return Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            compute_metrics=self.compute_metrics_fn
        )

    def train(self):
        self.trainer.train()
    
    def evaluate(self, dataset=None):
        self.trainer.evaluate(dataset)

    def get_default_codet5_hyperparameters(self):
        hyperparameters_config = OmegaConf.load(get_abs_path(self.DEFAULT_CODET5_HYPERPARAMETERS)).hyperparameters

        training_args = TrainingArguments(
            per_device_train_batch_size=hyperparameters_config["per_device_train_batch_size"],
            gradient_accumulation_steps=hyperparameters_config["gradient_accumulation_steps"],
            num_train_epochs=hyperparameters_config["num_train_epochs"],
            warmup_steps=hyperparameters_config["warmup_steps"],
            learning_rate=hyperparameters_config["learning_rate"],
            fp16=hyperparameters_config["fp16"],
            fsdp=hyperparameters_config["fsdp"],
            sharded_ddp=hyperparameters_config["sharded_ddp"],
            logging_steps=hyperparameters_config["logging_steps"],
            evaluation_strategy=hyperparameters_config["evaluation_strategy"],
            gradient_checkpointing=hyperparameters_config["gradient_checkpointing"],
            auto_find_batch_size=hyperparameters_config["auto_find_batch_size"],
            output_dir=self.checkpoints_path
        )
        # return hyperparameters_config
        return training_args
    
    def get_default_causal_lm_hyperparameters(self):
        hyperparameters_config = OmegaConf.load(get_abs_path(self.DEFAULT_CAUSAL_LM_HYPERPARAMETERS)).hyperparameters

        training_args = TrainingArguments(
            per_device_train_batch_size=hyperparameters_config["per_device_train_batch_size"],
            gradient_accumulation_steps=hyperparameters_config["gradient_accumulation_steps"],
            num_train_epochs=hyperparameters_config["num_train_epochs"],
            warmup_steps=hyperparameters_config["num_train_epochs"],
            learning_rate=hyperparameters_config["learning_rate"],
            fp16=hyperparameters_config["fp16"],
            fsdp=hyperparameters_config["fsdp"],
            sharded_ddp=hyperparameters_config["sharded_ddp"],
            logging_steps=hyperparameters_config["logging_steps"],
            evaluation_strategy=hyperparameters_config["evaluation_strategy"],
            gradient_checkpointing=hyperparameters_config["gradient_checkpointing"],
            auto_find_batch_size=hyperparameters_config["auto_find_batch_size"],
            output_dir=self.checkpoints_path
        )
        # return hyperparameters_config
        return training_args
    
    def get_default_lora_config_for_codet5(self):
        
        config = OmegaConf.load(get_abs_path(self.DEFAULT_CODET5_HYPERPARAMETERS)).lora

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            inference_mode=False, r=config["r"], 
            lora_alpha=config["lora_alpha"], 
            lora_dropout=config["lora_dropout"]
        )
        return lora_config

    # def get_default_adalora_config_for_codet5(self):
        
    #     config = OmegaConf.load(get_abs_path(self.PEFT_CODET5_CONFIGS)).adalora

    #     adalora_config = AdaLoraConfig(
    #         init_r=config["init_r"],
    #         target_r=config["target_r"],
    #         beta1=config["beta1"],
    #         beta2=config["beta2"],
    #         tinit=config["tinit"],
    #         tfinal=config["tfinal"],
    #         deltaT=config["deltaT"],
    #         lora_alpha=config["lora_alpha"],
    #         lora_dropout=config["lora_dropout"],
    #         task_type=TaskType.SEQ_2_SEQ_LM,
    #         inference_mode=False,
    #     )

    #     return adalora_config

    def create_checkpoints_path(self, checkpoints_path):
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
       
    def get_default_prefixtuning_config_for_codet5(self):
        
        config = OmegaConf.load(get_abs_path(self.DEFAULT_CODET5_HYPERPARAMETERS)).prefixtuning

        prefixtuning_config = PrefixTuningConfig(
            num_virtual_tokens=config["num_virtual_tokens"],
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        return prefixtuning_config
    
    

       
    
   