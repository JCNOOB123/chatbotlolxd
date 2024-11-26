import torch
from model.transformer import TransformerModel
from model.tokenizer import SimpleTokenizer
from model.train import ChatDataset, train_model
from torch.utils.data import DataLoader

# Sample training data
conversations = [
    {"input": "Hello", "response": "Hi there! How can I help you?"},
    {"input": "How are you?", "response": "I'm doing well, thank you for asking!"},
    {"input": "What's your name?", "response": "I'm an AI assistant created to help you."},
    {"input": "What can you do?", "response": "I can chat with you and help answer your questions."},
    # Add more training examples here
]

# Initialize tokenizer and model
tokenizer = SimpleTokenizer()
model = TransformerModel(
    vocab_size=tokenizer.vocab_size,
    d_model=256,
    nhead=8
)

# Create dataset and dataloader
dataset = ChatDataset(conversations, tokenizer)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train for 10 epochs
train_model(model, train_loader, num_epochs=10, device=device)

# Save the model
torch.save(model.state_dict(), 'chatbot_model.pth') 