import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.wq = nn.Linear(d_model, d_model, bias = False)
        self.wk = nn.Linear(d_model, d_model, bias = False)
        self.wv = nn.Linear(d_model, d_model, bias = False)
        self.m = torch.tril(torch.ones(max_seq_len, max_seq_len))
        #print("here",self.m)
        mask = self.m.masked_fill(self.m==0,float("-inf"))
        mask = mask.masked_fill(mask==1, 0)
        #print("making", self.mask)
        self.register_buffer('mask', mask)
    
    def forward(self, x, padding_mask=None):
        q = self.wq(x) # batch_size x sequence_length x emb_size
        k = self.wk(x) # batch_size x sequence_length x emb_size
        v = self.wv(x) # batch_size x sequence_length x emb_size
        # print("q is", q)
        # print("k is", k)
        # print("v is", v)
        attn_scores = (q @ k.transpose(-2,-1))/math.sqrt(self.d_model)
        # print(attn_scores)
        
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(padding_mask==0, float("-inf"))
        seq_len = x.shape[1]
        attn_masked = attn_scores + self.mask[:seq_len, :seq_len].unsqueeze(0)
        # print("masked", attn_masked)
        attn_soft = torch.softmax(attn_masked, dim=-1)
        attn_soft = torch.nan_to_num(attn_soft, nan=0.0)
        # print(attn_soft)
        final_attn = attn_soft @ v
        return final_attn

# testing
# x = torch.tensor([[
#     [2, -1, 3, -2],
#     [1, 2, -1, 1.5],
#     [-1, 0.5, 2, -0.5]
# ]])

# padding = torch.tensor([[
#     1,1,0
# ]])

# a = Attention(4, 3)
# output = a(x, padding)
# print(output)