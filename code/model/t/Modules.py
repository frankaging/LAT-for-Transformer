import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Zhengxuan Wu"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            w_len = attn.shape[-1]
            mask_t = mask.transpose(2,3).contiguous()
            mask_t = torch.cat(w_len*[mask_t], dim=2)
            attn = attn.masked_fill(mask_t == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)

        return output, attn
