import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformer import TransformerModel
from tokenizer import SimpleTokenizer
import torch.optim as optim

class ChatDataset(Dataset):
    def __init__(self, conversations, tokenizer):
        self.conversations = conversations
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        input_text = conv['input']
        target_text = conv['response']
        
        input_ids = self.tokenizer.encode(input_text)
        target_ids = self.tokenizer.encode(target_text)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }

def train_model(model, train_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Create masks
            src_mask = model.transformer.generate_square_subsequent_mask(input_ids.size(1)).to(device)
            tgt_mask = model.transformer.generate_square_subsequent_mask(target_ids.size(1)).to(device)
            
            # Forward pass
            outputs = model(input_ids, target_ids, src_mask, tgt_mask)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}') 