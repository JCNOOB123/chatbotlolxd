import torch
from transformer import TransformerModel
from tokenizer import SimpleTokenizer

class ChatBot:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = SimpleTokenizer()
        self.model = TransformerModel(
            vocab_size=self.tokenizer.vocab_size,
            d_model=256,
            nhead=8
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
        self.device = device
        
    def generate_response(self, input_text, max_length=50):
        input_ids = self.tokenizer.encode(input_text).to(self.device)
        
        # Initialize target with start token
        target_ids = torch.tensor([[self.tokenizer.tokenizer.bos_token_id]], device=self.device)
        
        for _ in range(max_length):
            src_mask = self.model.transformer.generate_square_subsequent_mask(input_ids.size(1)).to(self.device)
            tgt_mask = self.model.transformer.generate_square_subsequent_mask(target_ids.size(1)).to(self.device)
            
            output = self.model(input_ids, target_ids, src_mask, tgt_mask)
            next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)
            target_ids = torch.cat([target_ids, next_token], dim=1)
            
            if next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                break
                
        response = self.tokenizer.decode(target_ids[0])
        return response 