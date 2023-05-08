

import torch
import torch.nn.functional as F
from datasets import load_dataset
 from accelerate import Accelerator
 
class CodeT5Seq2SeqTrainer():
    
    def __init__(self, from_pretrained=None):
        # super().__init__(model, max_source_length, max_target_length, beam_size, tokenizer_path)
        
        self.model = T5ForConditionalGeneration.from_pretrained(from_pretrained)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.codet5_model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

    # def get_default_hyperparameters():
    #     self.max_source

    def forward(self, sources, targets):
        # input_ids = self.tokenizer(sources, padding=True, return_tensors='pt').input_ids.to(self.device)
        # target_ids = self.tokenizer(targets, padding=True, return_tensors='pt').input_ids.to(self.device)

        generated_ids = self.model(sources, targets)
        loss = self.criterion(generated_ids.view(-1, generated_ids.size(-1)), target_ids.view(-1))
        
        return loss

    def train(self, sources, targets):
        for epoch in range(self.num_epochs):
            self.model.train()
            loss = self.forward(sources, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            print('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.num_epochs, loss.item()))
    

    # def evaluate(self, sources, targets):
    #     for epoch in range(self.num_epochs):
    #         self.codet5_model.train()
    #         loss = self.forward(sources, targets)
            
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
            
    #         print('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.num_epochs, loss.item()))




