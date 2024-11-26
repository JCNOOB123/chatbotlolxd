from transformers import GPT2Tokenizer

class SimpleTokenizer:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors='pt')
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size 