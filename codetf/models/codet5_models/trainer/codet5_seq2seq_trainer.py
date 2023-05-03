

class CodeT5Seq2SeqTrainer(CodeT5BaseModel):
    def __init__(self, model, max_source_length, max_target_length, beam_size, tokenizer_path, learning_rate, num_epochs):
        super().__init__(model, max_source_length, max_target_length, beam_size, tokenizer_path)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.codet5_model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs

    def forward(self, sources, targets):
        input_ids = self.tokenizer(sources, padding=True, return_tensors='pt').input_ids.to(self.device)
        target_ids = self.tokenizer(targets, padding=True, return_tensors='pt').input_ids.to(self.device)

        generated_ids = self.codet5_model(input_ids, target_ids)
        loss = self.criterion(generated_ids.view(-1, generated_ids.size(-1)), target_ids.view(-1))
        
        return loss

    def train(self, sources, targets):
        for epoch in range(self.num_epochs):
            self.codet5_model.train()
            loss = self.forward(sources, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            print('Epoch: [{}/{}], Loss: {:.4f}'.format(epoch + 1, self.num_epochs, loss.item()))




