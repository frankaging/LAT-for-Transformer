''' Define the Layers '''
import torch.nn as nn
import torch
from t.SubLayers import MultiHeadAttention, PositionwiseFeedForward


__author__ = "Zhengxuan Wu"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn, q_ma_last_pre, q_ma_last_post_ret, \
        attn_pre, attn_post, \
        v_ma_first_pre, v_ma_first_post = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output, x_1_pre, x_1_post, x_2_pre, x_2_post = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn, x_1_pre, x_1_post, x_2_pre, x_2_post, \
                q_ma_last_pre, q_ma_last_post_ret, \
                attn_pre, attn_post, \
                v_ma_first_pre, v_ma_first_post
