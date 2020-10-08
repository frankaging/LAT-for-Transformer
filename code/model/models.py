from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

from t.Models import *
from make_masks import *

def pad_shift(x, shift, padv=0.0):
    """Shift 3D tensor forwards in time with padding."""
    if shift > 0:
        padding = torch.ones(x.size(0), shift, x.size(2)).to(x.device) * padv
        return torch.cat((padding, x[:, :-shift, :]), dim=1)
    elif shift < 0:
        padding = torch.ones(x.size(0), -shift, x.size(2)).to(x.device) * padv
        return torch.cat((x[:, -shift:, :], padding), dim=1)
    else:
        return x

def convolve(x, attn):
    """Convolve 3D tensor (x) with local attention weights (attn)."""
    stacked = torch.stack([pad_shift(x, i) for
                           i in range(attn.shape[2])], dim=-1)
    return torch.sum(attn.unsqueeze(2) * stacked, dim=-1)

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class TransformerLinearAttn(nn.Module):
    '''
    Model Code: bd01a5fa-07d4-4870-8ef8-303abd397874

    Self-attention + context vector (full attention) based encoder + linear decoder.
    This model is for the Stanford Sentiment Treebank. This implementation served 
    as the purpose of reproducing the interpretatable attention propagation developed
    on the Stanford Emotional Narratives Datasets. More info is shown below 
    @TransformerLSTMAttn.
    '''

    def __init__(self, mods, dims,
                 device=torch.device('cuda:0')):
        super(TransformerLinearAttn, self).__init__()
        # init
        self.mods = mods
        self.dims = dims
        self.window_embed_size={'linguistic' : 300}

        # self-attention window embeddings
        self.att_n_layer = 6
        self.att_n_header = 8
        self.encoder_in = self.window_embed_size['linguistic']
        self.encoder_out = 300
        att_d_k = 64
        self.att_d_v = 64
        att_d_model = self.encoder_out
        att_d_inner = 64
        self.attendedEncoder = attendedEncoder(self.att_n_layer,
                                               self.att_n_header,
                                               att_d_k,
                                               self.att_d_v,
                                               att_d_model,
                                               att_d_inner)

        # attention gate
        attn_dropout = 0.1
        self.dropout = nn.Dropout(attn_dropout)
        self.encoder_gate = nn.Sequential(nn.Linear(self.encoder_out, 128),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(64, 1))

        # final output layers
        final_out = self.encoder_out
        h_out = 64
        output_dim = 5
        out_dropout = 0.3
        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)
        self.out_act = nn.Sigmoid()

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, length, mask=None):
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        # self-attention layers
        mask_bool = mask.bool()
        mask_bool = mask_bool.unsqueeze(dim=-1)
        # transformer encoder
        attended_out, _,_,_,_,_,_,_,_,_,_,_ = \
            self.attendedEncoder(inputs,
                                 mask_bool)

        # context layer
        attn = self.encoder_gate(attended_out)
        attn = attn.masked_fill(mask_bool == 0, -1e9)
        attn = F.softmax(attn, dim=1)
        # attened embeddings
        hs_attend = \
            torch.matmul(attn.permute(0,2,1), attended_out).squeeze(dim=1)

        # final output blocks
        hs_tw_fc1_pre = hs_attend.clone()
        hs_tw_fc1 = self.out_fc1(hs_attend)
        hs_tw_fc1_post = hs_tw_fc1.clone()
        hs_tw_fc2 = self.out_dropout(F.relu(hs_tw_fc1))
        hs_tw_fc2_pre = hs_tw_fc2.clone()
        hs_tw_fc2_post = self.out_fc2(hs_tw_fc2)
        target_pre = hs_tw_fc2_post
        target_ret = target_pre.clone()
        target = self.out_act(target_pre)

        return target

    def backward_nlap(self, inputs, length, mask=None):
        '''
        This is backing out the attention using the context based attention and
        the attentions within the transformer using naive lap method proposed.
        '''
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        # self-attention layers
        mask_bool = mask.bool()
        mask_bool = mask_bool.unsqueeze(dim=-1)
        # transformer encoder
        attended_out, tf_attns,_,_,_,_,_,_,_,_,_,_ = \
            self.attendedEncoder(inputs,
                                 mask_bool)

        # context layer
        attn = self.encoder_gate(attended_out)
        attn = attn.masked_fill(mask_bool == 0, -1e9)
        attn = F.softmax(attn, dim=1)

        # self attention backout
        tf_attns = torch.stack(tf_attns, dim=0).permute(2,0,1,3,4)
        raw_attns = []
        for h in range(tf_attns.shape[0]):
            tf_attn = tf_attns[h]
            pre_attn = attn.clone().permute(0, 2, 1)
            for i in reversed(range(self.att_n_layer)):
                curr_tf_attn = torch.matmul(pre_attn, tf_attn[i])
                pre_attn = curr_tf_attn
            raw_attns.append(pre_attn.permute(0,2,1))

        raw_attns = torch.stack(raw_attns, dim=0).sum(dim=0)
        return raw_attns.squeeze(dim=-1)

    def backward_tf_attn(self, inputs, length, mask=None):
        '''
        This is returning the transformer attention for each layer and each head.
        '''
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        # self-attention layers
        mask_bool = mask.bool()
        mask_bool = mask_bool.unsqueeze(dim=-1)
        # transformer encoder
        attended_out, tf_attns,_,_,_,_,_,_,_,_,_,_ = \
            self.attendedEncoder(inputs,
                                 mask_bool)
        # context layer
        attn = self.encoder_gate(attended_out)
        attn = attn.masked_fill(mask_bool == 0, -1e9)
        attn = F.softmax(attn, dim=1)

        ctx_attn = attn.clone().squeeze(dim=-1)
        tf_attns = torch.stack(tf_attns, dim=0).permute(1,2,0,3,4).contiguous()

        return tf_attns, ctx_attn

class TransformerLSTMAttn(nn.Module):
    '''
    Model Code: df5f97d3-90eb-40e0-8c85-ae6218c20d1e

    Self-attention + context vector (full attention) based encoder + LSTM decoder.
    This model is for the Stanford Emotional Narratives Datasets. If you want to
    use the dataset, considering use and cite this repo:
    https://github.com/StanfordSocialNeuroscienceLab/SEND

    With in model, it contains 4 different ways of backing out attentions in the 
    model:

    1. ctx_nlap: this is backing out attentions only on context layer and without 
    directional encoding.

    2. nlap: this is backing out attentions using both context layer and self-attentions
    but it is without directional encoding.
    '''

    def __init__(self, mods, dims,
                 device=torch.device('cuda:0')):
        super(TransformerLSTMAttn, self).__init__()
        # init
        self.mods = mods
        self.dims = dims
        self.window_embed_size={'linguistic' : 300}

        # self-attention window embeddings
        self.att_n_layer = 6
        self.att_n_header = 8
        self.encoder_in = self.window_embed_size['linguistic']
        self.encoder_out = 300
        att_d_k = 64
        self.att_d_v = 64
        att_d_model = self.encoder_out
        att_d_inner = 64
        self.attendedEncoder = attendedEncoder(self.att_n_layer,
                                               self.att_n_header,
                                               att_d_k,
                                               self.att_d_v,
                                               att_d_model,
                                               att_d_inner)

        # attention gate
        attn_dropout = 0.1
        self.dropout = nn.Dropout(attn_dropout)
        self.encoder_gate = nn.Sequential(nn.Linear(self.encoder_out, 128),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(64, 1))

        # second layer recurrent network
        self.rnn_in = self.encoder_out
        self.rnn_out = self.encoder_out
        self.rnn_layers = 1
        self.rnn_tw = nn.LSTM(self.rnn_in,
                              self.rnn_out,
                              self.rnn_layers,
                              batch_first=True)

        # final output layers
        final_out = self.encoder_out
        h_out = 64
        output_dim = 1
        out_dropout = 0.3
        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)

        # Store module in specified device (CUDA/CPU)
        self.device = (device if torch.cuda.is_available() else
                       torch.device('cpu'))
        self.to(self.device)

    def forward(self, inputs, length, token_length, mask=None):
        '''
        inputs = dict{} of (batch_size, seq_len, dim)
        '''
        # set the input to only single channel
        single_mod = inputs['linguistic']

        # generate token mask for encoder to use (only for att model)
        global_max_token_length = single_mod.shape[2]
        token_mask, token_len_pad = generate_token_mask(length, token_length,
                                                        global_max_token_length,
                                                        self.device)

        # params
        batch_size = len(length)
        assert(batch_size == single_mod.shape[0])
        max_len = max(length)
        assert(max_len == single_mod.shape[1])
        max_token = token_mask.shape[-1]

        # reshape
        single_mod_flat = single_mod.reshape(batch_size*max_len, max_token, self.encoder_in)
        token_len_pad_flat = token_len_pad.reshape(batch_size*max_len,)
        token_mask_flat = token_mask.reshape(batch_size*max_len, -1)

        # transformer encoder
        attended_out, _,_,_,_,_,_,_,_,_,_,_ = \
            self.attendedEncoder(single_mod_flat,
                                 token_mask_flat.unsqueeze(dim=-1))

        # get gated attention
        attn = self.encoder_gate(attended_out)
        token_mask_flat = token_mask_flat.unsqueeze(dim=-1)
        attn = attn.masked_fill(token_mask_flat == 0, -1e9)
        attn = F.softmax(attn, dim=1)

        # attened embeddings
        hs_attend = \
            torch.matmul(attn.permute(0,2,1), attended_out).squeeze(dim=1)
        hs_attend = hs_attend.reshape(batch_size, max_len, -1)

        # rnn on time windows
        embed_tw = pack_padded_sequence(hs_attend, length,
                                        batch_first=True,
                                        enforce_sorted=False)
        h0_tw = torch.zeros(self.rnn_layers, batch_size,
                            self.encoder_out).to(self.device)
        c0_tw = torch.zeros(self.rnn_layers, batch_size,
                            self.encoder_out).to(self.device)
        hs_tw, _ = self.rnn_tw(embed_tw, (h0_tw, c0_tw))
        hs_tw, _ = pad_packed_sequence(hs_tw, batch_first=True) # hs: (b*l,tl,n)

        # single feed-forward layers for the final outpus
        target = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(hs_tw))))
        target = target * mask.float()

        return target

    def backward_nlap_ctx(self, inputs, length, token_length, mask=None):
        '''
        This is backing out the attention only using the context based attention
        scores.
        '''
        # set the input to only single channel
        single_mod = inputs['linguistic']

        # generate token mask for encoder to use (only for att model)
        global_max_token_length = single_mod.shape[2]
        token_mask, token_len_pad = generate_token_mask(length, token_length,
                                                        global_max_token_length,
                                                        self.device)

        # params
        batch_size = len(length)
        assert(batch_size == single_mod.shape[0])
        max_len = max(length)
        assert(max_len == single_mod.shape[1])
        max_token = token_mask.shape[-1]

        # reshape
        single_mod_flat = single_mod.reshape(batch_size*max_len, max_token, self.encoder_in)
        token_len_pad_flat = token_len_pad.reshape(batch_size*max_len,)
        token_mask_flat = token_mask.reshape(batch_size*max_len, -1)

        # transformer encoder
        attended_out, tf_attns, _,_,_,_,_,_,_,_,_,_ = \
            self.attendedEncoder(single_mod_flat,
                                 token_mask_flat.unsqueeze(dim=-1))

        # get gated attention
        attn = self.encoder_gate(attended_out)
        token_mask_flat = token_mask_flat.unsqueeze(dim=-1)
        attn = attn.masked_fill(token_mask_flat == 0, -1e9)
        attn = F.softmax(attn, dim=1)

        return attn.squeeze(dim=-1)

    def backward_nlap(self, inputs, length, token_length, mask=None):
        '''
        This is backing out the attention using the context based attention and
        the attentions within the transformer using naive lap method proposed.
        '''
        # set the input to only single channel
        single_mod = inputs['linguistic']

        # generate token mask for encoder to use (only for att model)
        global_max_token_length = single_mod.shape[2]
        token_mask, token_len_pad = generate_token_mask(length, token_length,
                                                        global_max_token_length,
                                                        self.device)

        # params
        batch_size = len(length)
        assert(batch_size == single_mod.shape[0])
        max_len = max(length)
        assert(max_len == single_mod.shape[1])
        max_token = token_mask.shape[-1]

        # reshape
        single_mod_flat = single_mod.reshape(batch_size*max_len, max_token, self.encoder_in)
        token_len_pad_flat = token_len_pad.reshape(batch_size*max_len,)
        token_mask_flat = token_mask.reshape(batch_size*max_len, -1)

        # transformer encoder
        attended_out, tf_attns, _,_,_,_,_,_,_,_,_,_ = \
            self.attendedEncoder(single_mod_flat,
                                 token_mask_flat.unsqueeze(dim=-1))

        # get gated attention
        attn = self.encoder_gate(attended_out)
        token_mask_flat = token_mask_flat.unsqueeze(dim=-1)
        attn = attn.masked_fill(token_mask_flat == 0, -1e9)
        attn = F.softmax(attn, dim=1)

        # self attention backout
        tf_attns = torch.stack(tf_attns, dim=0).permute(2,0,1,3,4)
        raw_attns = []
        for h in range(tf_attns.shape[0]):
            tf_attn = tf_attns[h]
            pre_attn = attn.permute(0, 2, 1)
            for i in reversed(range(self.att_n_layer)):
                curr_tf_attn = torch.matmul(pre_attn, tf_attn[i])
                pre_attn = curr_tf_attn
            raw_attns.append(pre_attn.permute(0,2,1))

        raw_attns = torch.stack(raw_attns, dim=0).sum(dim=0)
        return raw_attns.squeeze(dim=-1)

    def backward_lap_ctx(self, inputs, length, token_length, mask=None):
        '''
        This is backing out the attention using only the context based attention
        using lap method proposed.
        '''
        # set the input to only single channel
        single_mod = inputs['linguistic']

        # generate token mask for encoder to use (only for att model)
        global_max_token_length = single_mod.shape[2]
        token_mask, token_len_pad = generate_token_mask(length, token_length,
                                                        global_max_token_length,
                                                        self.device)

        # params
        batch_size = len(length)
        assert(batch_size == single_mod.shape[0])
        max_len = max(length)
        assert(max_len == single_mod.shape[1])
        max_token = token_mask.shape[-1]

        # reshape
        single_mod_flat = single_mod.reshape(batch_size*max_len, max_token, self.encoder_in)
        token_len_pad_flat = token_len_pad.reshape(batch_size*max_len,)
        token_mask_flat = token_mask.reshape(batch_size*max_len, -1)

        # transformer encoder
        attended_out, tf_attns, \
        x_1_pre_list, x_1_post_list, x_2_pre_list, x_2_post_list, \
        q_ma_last_pre_list, q_ma_last_post_list, \
        attn_pre_list, attn_post_list, \
        v_ma_first_pre_list, v_ma_first_post_list = \
            self.attendedEncoder(single_mod_flat,
                                 token_mask_flat.unsqueeze(dim=-1))

        # get gated attention
        attn = self.encoder_gate(attended_out)
        token_mask_flat = token_mask_flat.unsqueeze(dim=-1)
        attn = attn.masked_fill(token_mask_flat == 0, -1e9)
        attn = F.softmax(attn, dim=1)

        # attened embeddings
        hs_attend = \
            torch.matmul(attn.permute(0,2,1), attended_out).squeeze(dim=1)
        hs_attend = hs_attend.reshape(batch_size, max_len, -1)

        # rnn on time windows
        embed_tw = pack_padded_sequence(hs_attend, length,
                                        batch_first=True,
                                        enforce_sorted=False)
        h0_tw = torch.zeros(self.rnn_layers, batch_size,
                            self.encoder_out).to(self.device)
        c0_tw = torch.zeros(self.rnn_layers, batch_size,
                            self.encoder_out).to(self.device)
        hs_tw, _ = self.rnn_tw(embed_tw, (h0_tw, c0_tw))
        hs_tw, _ = pad_packed_sequence(hs_tw, batch_first=True) # hs: (b*l,tl,n)

        # single feed-forward layers for the final outpus
        hs_tw_fc1_post = self.out_fc1(hs_tw)
        hs_tw_fc2_pre = self.out_dropout(F.relu(hs_tw_fc1_post))
        hs_tw_fc2_post = self.out_fc2(hs_tw_fc2_pre)
        target = hs_tw_fc2_post * mask.float()

        # record the time for videos
        start = time.time()

        print("Starting to back out for a video with window size: " + str(target.shape[1]))

        ######################################################################
        #
        # Backout from the softmax context layer
        #
        ######################################################################

        # get lap based attn scores
        post_hs = hs_attend.squeeze(dim=0).unsqueeze(dim=1) # always single seq per batch
        pre_hs = attended_out
        attn_in = attn.transpose(2,1)


        post_A = target.squeeze(dim=0).clone()
        # clip to -1 and 1; and then turn into revelant score
        post_A = torch.stack(post_hs.shape[2]*[post_A], dim=-1)

        lap_b = post_hs.shape[0]
        attn_lap = []

        for i in range(lap_b):

            # backout from context layer
            pre_A_i = lap(post_hs[i], pre_hs[i], attn_in[i], post_A[i])

            # save this window attention backout results
            post_A_l_ret = pre_A_i.clone()
            attn_lap.append(post_A_l_ret)

        end = time.time()
        print("Time elapse (sec): " + str((end-start)))

        # summarize by taking sum of lap for each vector
        attn_lap = torch.stack(attn_lap, dim=0).sum(dim=-1)
        return attn_lap

    def backward_tf_attn(self, inputs, length, token_length, mask=None):
        '''
        This is returning the transformer attention for each layer and each head.
        '''
        # set the input to only single channel
        single_mod = inputs['linguistic']

        # generate token mask for encoder to use (only for att model)
        global_max_token_length = single_mod.shape[2]
        token_mask, token_len_pad = generate_token_mask(length, token_length,
                                                        global_max_token_length,
                                                        self.device)

        # params
        batch_size = len(length)
        assert(batch_size == single_mod.shape[0])
        max_len = max(length)
        assert(max_len == single_mod.shape[1])
        max_token = token_mask.shape[-1]

        # reshape
        single_mod_flat = single_mod.reshape(batch_size*max_len, max_token, self.encoder_in)
        token_len_pad_flat = token_len_pad.reshape(batch_size*max_len,)
        token_mask_flat = token_mask.reshape(batch_size*max_len, -1)

        # transformer encoder
        attended_out, tf_attns, \
        x_1_pre_list, x_1_post_list, x_2_pre_list, x_2_post_list, \
        q_ma_last_pre_list, q_ma_last_post_list, \
        attn_pre_list, attn_post_list, \
        v_ma_first_pre_list, v_ma_first_post_list = \
            self.attendedEncoder(single_mod_flat,
                                 token_mask_flat.unsqueeze(dim=-1))

        # get gated attention
        attn = self.encoder_gate(attended_out)
        token_mask_flat = token_mask_flat.unsqueeze(dim=-1)
        attn = attn.masked_fill(token_mask_flat == 0, -1e9)
        attn = F.softmax(attn, dim=1)

        ctx_attn = attn.clone().squeeze(dim=-1)
        tf_attns = torch.stack(tf_attns, dim=0).permute(1,2,0,3,4).contiguous()

        return tf_attns, ctx_attn
