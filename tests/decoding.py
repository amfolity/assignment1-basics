import torch
from .bpe import Tokenizer
from .model import TransformerLM

config = {
           'lr': 0.01,
           'batch_size' : 8, 
            'steps':1000,
    'context_length': 64, 
        }
  

def decoding(prompt, tokenizer, model, max_new_tokens=200, top_p=0.9, temp=0.8):
    #tokenizer = Tokenizer(vocab, merges)

    input_ids = tokenizer.encode(prompt)
    
    #lm = TransformerLM(vocab_size, context_length, num_layers, d_model, d_ff, num_heads)   

    for _ in range(max_new_tokens - 1):
        logits = lm(input_ids)
        sorted_logits = torch.sort(logits, descending=True)
        
        probs = F.softmax(sorted_logits / temp, dim=-1)
        cumulative_probs = torch.cumsum(sorted_logits)

        mask = cumulative_probs < top_p
        sorted_logits[mask] = float('-inf')

        filtered_logits = torch.zeros_like(scaled_logits)
        new_probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == "<|endoftext|>":
            break

    return intput_ids
            
        
        
    