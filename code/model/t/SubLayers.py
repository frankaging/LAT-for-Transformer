''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from t.Modules import ScaledDotProductAttention
import torch

__author__ = "Zhengxuan Wu"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)

        # ops on v to save all the context for backing out
        v_ma_first_pre = v.clone()
        v_ma_first_post = self.w_vs(v)
        v_ma_first_post_ret = v_ma_first_post.clone()
        v = v_ma_first_post.view(sz_b, len_v, n_head, d_v)
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        attn_pre = v.clone()

        q_hs, attn = self.attention(q, k, v, mask=mask)

        attn_post = q_hs.clone() # b x n x lq x dv

        q_hs = q_hs.transpose(1, 2).contiguous()

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q_ma_last_pre = q_hs.view(sz_b, len_q, -1)
        q_ma_last_post = self.fc(q_ma_last_pre)
        q_ma_last_post_ret = q_ma_last_post.clone()
        q = self.dropout(q_ma_last_post)
        q += residual

        return q, attn, q_ma_last_pre, q_ma_last_post_ret, attn_pre, attn_post, \
                v_ma_first_pre, v_ma_first_post_ret


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x_1_pre = self.layer_norm(x)
        x_1_post = self.w_1(x_1_pre)

        x_2_pre = F.relu(x_1_post)
        x_2_post = self.w_2(x_2_pre)

        x_2_post_ret = x_2_post.clone()

        x = self.dropout(x_2_post)
        x += residual

        return x, x_1_pre, x_1_post, x_2_pre, x_2_post_ret
