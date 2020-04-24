import torch
import torch.nn as nn
import numpy as np

# These functions are helpers to generate different kinds of masks.

def get_attention_mask(length, mask, device, attention_len=10):
    batch_size, seq_len = len(length), max(length)
    ori_mask = mask.squeeze(dim=-1)
    batch_mask = []
    for i in range(batch_size):
        seq_mask = []
        for j in range(seq_len):
            m = torch.zeros(seq_len).to(device)
            if j <  length[i]:
                if j < attention_len:
                    m[:attention_len,] = 1
                else:
                    m[j-attention_len+1:j+1,] = 1
            m = m.bool() & ori_mask[i].bool()
            seq_mask.append(m)
        seq_mask = torch.stack(seq_mask, dim=0)
        batch_mask.append(seq_mask)
    batch_mask = torch.stack(batch_mask, dim=0)
    return batch_mask

def xstransformer_gs(attended_scores):
    # consolitation layers weights
    attended_scores = \
        torch.stack(attended_scores, dim=0)
    # only single sequence for backward
    attended_scores = attended_scores.squeeze(dim=1)
    att_n_layer = attended_scores.shape[0]
    att_n_head = attended_scores.shape[1]
    seq_len = attended_scores.shape[-1]

    prev_attended_score = torch.eye(seq_len)
    prev_attended_score = torch.stack(att_n_head*[torch.eye(seq_len)], dim=0)

    for i in range(att_n_layer):
        curr_attended_score = torch.bmm(prev_attended_score, attended_scores[att_n_layer - i - 1])
        prev_attended_score = curr_attended_score
    return prev_attended_score

def generate_token_mask(length, token_length, global_max_token_length, device, ret_bool=True):
    '''
    WARNING: We always think at least 1 token is effective in 1 seq.
    '''
    batch_size = len(length)
    max_len = max(length)
    max_token_len = global_max_token_length

    token_mask = torch.zeros((batch_size, max_len, max_token_len)).to(device)
    token_mask[:,:,0] = 1.0
    for b in range(batch_size):
        for t in range(length[b]):
            token_mask[b,t,:token_length[b][t]] = 1.

    token_len_pad = torch.ones((batch_size, max_len)).to(device)
    for b in range(batch_size):
        for t in range(length[b]):
            token_len_pad[b,t] = token_length[b][t]

    if ret_bool:
        return token_mask.bool(), token_len_pad
    else:
        return token_mask, token_len_pad