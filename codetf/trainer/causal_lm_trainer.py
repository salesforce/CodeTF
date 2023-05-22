from transformers import AutoModelForCausalLM
from codetf.trainer.base_trainer import BaseTrainer

class CausalLMTrainer(BaseTrainer):    
    def __init__(self, train_dataset, validation_dataset=None, tokenizer=None, 
                checkpoints_path="./checkpoints", pretrained_model_or_path="gpt2", 
                training_args=None, evaluator=None, evaluation_fn=None, peft=None):
        
        # model = AutoModelForCausalLM.from_pretrained(pretrained_model_or_path)
        
       
            
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