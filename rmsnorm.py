import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        #print("how x looks",x)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        #print("this is rms", rms)
        #print(self.weights)
        output = (x / rms) * self.weights
        return output

#testing
# x = torch.tensor([
#     [2, -1, 3, -2],
#     [1, 2, -1, 1.5],
#     [-1, 0.5, 2, -0.5]
# ])

# norm = RMSNorm(4)
# output = norm(x)
# print(output.shape)