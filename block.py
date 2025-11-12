import torch
import torch.nn as nn
from attention import Attention
from ffn import FFN
from rmsnorm import RMSNorm

class Block(nn.Module):
    def __init__(self, hidden_dim, d_model, max_seq_len):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attention = Attention(d_model, max_seq_len)
        self.ffn = FFN(d_model, hidden_dim, d_model)
    
    def forward(self, x, padding_mask=None):
        x = x + self.attention(x, padding_mask)
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

#testing
# x = torch.tensor([[
#     [2, -1, 3, -2],
#     [1, 2, -1, 1.5],
#     [-1, 0.5, 2, -0.5]
# ]])

# a = Block(768, 4, 10)
# output = a(x)
# print(output.shape)