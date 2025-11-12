import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.seq_len = max_seq_len
        # for each sequence making a pe
        pe = torch.zeros(max_seq_len, d_model)
        # print("initial pe", self.pe)
        # tracking the positions of the sequence
        self.positions = torch.arange(0,max_seq_len).unsqueeze(1)
        self.dims = torch.arange(0, d_model, 2)
        self.div = torch.pow(10000, -2*self.dims/d_model)
        #print("positions shape", self.positions.shape, "div shape",self.div.shape)
        pe[:,::2] = torch.sin(self.positions * self.div)
        pe[:,1::2] = torch.cos(self.positions * self.div)

        self.register_buffer('pe', pe)

    def forward(self, x):
        #print("here is the unsqueezed one",self.pe.unsqueeze(0))
        p = self.pe[:x.size(-2),:] # adjust based on the sequence length
        return x + p.unsqueeze(0)
    
#testing
# x = torch.tensor([[
#     [2, -1, 3, -2],
#     [1, 2, -1, 1.5],
#     [-1, 0.5, 2, -0.5]
# ]])

# p = PositionalEncoding(10,4)
# output = p(x)
# print(output.shape)