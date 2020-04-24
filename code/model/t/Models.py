''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from t.Layers import EncoderLayer


__author__ = "Zhengxuan Wu"

# This will not be used in cross modal transformer
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class attendedEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, inputs, masks):

        enc_slf_attn_list = []

        x_1_pre_list = []
        x_1_post_list = []
        x_2_pre_list = []
        x_2_post_list = []

        q_ma_last_pre_list = []
        q_ma_last_post_list = []

        attn_pre_list = []
        attn_post_list = []

        v_ma_first_pre_list = []
        v_ma_first_post_list = []

        # -- Forward
        enc_output = self.dropout(inputs)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn, x_1_pre, x_1_post, x_2_pre, x_2_post, \
                q_ma_last_pre, q_ma_last_post_ret, \
                attn_pre, attn_post, \
                v_ma_first_pre, v_ma_first_post = enc_layer(enc_output, slf_attn_mask=masks)
            enc_slf_attn_list += [enc_slf_attn]

            x_1_pre_list += [x_1_pre]
            x_1_post_list += [x_1_post]
            x_2_pre_list += [x_2_pre]
            x_2_post_list += [x_2_post]

            q_ma_last_pre_list += [q_ma_last_pre]
            q_ma_last_post_list += [q_ma_last_post_ret]

            attn_pre_list += [attn_pre]
            attn_post_list += [attn_post]

            v_ma_first_pre_list += [v_ma_first_pre]
            v_ma_first_post_list += [v_ma_first_post]

        enc_output = self.layer_norm(enc_output)

        return enc_output, enc_slf_attn_list, \
                x_1_pre_list, x_1_post_list, x_2_pre_list, x_2_post_list, \
                q_ma_last_pre_list, q_ma_last_post_list, \
                attn_pre_list, attn_post_list, \
                v_ma_first_pre_list, v_ma_first_post_list