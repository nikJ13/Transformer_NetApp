import torch
import torch.nn as nn
from block import Block
from positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, hidden_dim, d_model, max_seq_len, vocab_size, num_blocks):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(hidden_dim, d_model, max_seq_len) for _ in range(num_blocks)])
        self.pos = PositionalEncoding(max_seq_len, d_model)
        self.linear_final = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, padding_mask = None):
        x = self.embd(x)
        x = self.pos(x)
        for b in self.blocks:
            x = b(x, padding_mask)
        x = self.linear_final(x)
        return x

#testing
# x = torch.tensor([[1,2,3]])
# print("HERE", x.shape)

# norm = Transformer(768, 4, 10, 512, 3)
# output = norm(x)
# print(output.shape)