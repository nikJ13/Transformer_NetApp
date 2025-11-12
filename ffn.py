import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.linear3(x)
        x = self.gelu(x)
        x = self.linear4(x)
        return x
    
# x = torch.tensor([[
#     [2, -1, 3, -2],
#     [1, 2, -1, 1.5],
#     [-1, 0.5, 2, -0.5]
# ]])

# a = FFN(4, 768, 4)
# output = a(x)
# print(output.shape)