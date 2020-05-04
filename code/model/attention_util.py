from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys, os, shutil
import argparse
import copy
import csv
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import seq_collate_dict, load_dataset
from models import *
from random import shuffle
import random
from operator import itemgetter
import pprint
from numpy import newaxis as na
import pickle

from string import punctuation
import statistics 
import nltk, re

'''
Visualization function, two parameters needed is just the sentence and the extracted attention weights.
This visualization function is for visualizing each head separately.
'''
def head_attn_viz_func(tf_attns, sentence, seq_id, save_fig=False):
    # input params
    tokens = tf_attns.shape[2]
    layers = 6
    heads = 8
    # generate points for making the connection plot
    point_offset = 0.3
    box_offset = 0.5
    layer_offset = point_offset * (tokens-1)

    # generate pair (of two points) and attention weight of the pair
    pair_weight = dict()
    pair_layer = dict()
    pair_head = dict()
    for i in range(heads):
        for j in range(layers):
            # for layer j, head i, we want to all the point pairs and weight
            base_x = i*(layer_offset + box_offset)
            base_y = j*(layer_offset + box_offset)
            # iteration through all the tokens to form pairs
            for k in range(tokens):
                k_x = base_x + k*(point_offset)
                k_y = base_y
                for l in range(tokens):
                    l_x = base_x + l*(point_offset)
                    l_y = base_y + layer_offset
                    pair_weight[((k_x, k_y),(l_x, l_y))] = \
                        tf_attns[i, j, l, k]
                    pair_layer[((k_x, k_y),(l_x, l_y))] = j
                    pair_head[((k_x, k_y),(l_x, l_y))] = i

    colors = ['red', 'chocolate', 'gold', 'greenyellow', 'darkgreen', 'aqua', 'blue', 'fuchsia']

    # plot all the points, and connect them with lines
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) # nrows, ncols, index
    for pair in pair_weight.keys():
        plt.plot([pair[0][0], pair[1][0]], 
                [pair[0][1], pair[1][1]], 
                'w-',
                linewidth=2.5,
                alpha=pair_weight[pair],
                color=colors[pair_head[pair]])
    ax.set_facecolor('black')
    plt.xticks([(i*(layer_offset + box_offset)+layer_offset*0.5) for i in range(heads)], ['0', '1', '2', '3', '4', '5', '6', '7'])
    plt.yticks([(i*(layer_offset + box_offset)+layer_offset*0.5) for i in range(layers)], ['0', '1', '2', '3', '4', '5'])
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.xlabel('Heads', fontsize=15)
    plt.ylabel('Layers', fontsize=15)
    ax.xaxis.set_label_position('top')
    ax.xaxis.label.set_color('grey')
    ax.yaxis.label.set_color('grey')

    # annotate with the text of this sentence
    annotate_str = "["
    for word in sentence:
        annotate_str += "\"" + word + "\"" + ", "
    annotate_str = annotate_str[:-2]
    annotate_str += "]"

    ax.annotate(annotate_str,  # Your string
                # The point that we'll place the text in relation to 
                xy=(0.5, 0), 
                # Interpret the x as axes coords, and the y as figure coords
                xycoords=('axes fraction', 'figure fraction'),
                # The distance from the point that the text will be at
                xytext=(0, 10),  
                # Interpret `xytext` as an offset in points...
                textcoords='offset points',
                # Any other text parameters we'd like
                size=14, ha='center', va='bottom',
                color='grey')
    # the output plot can be rescaled using this
    plt.show()

    if save_fig:
        plt.savefig('../tf_attns_plots/' + str(seq_id) + '.png', bbox_inches='tight')
        plt.close(fig)


'''
Visualization function for accumulative attention over all the layers. From the 
context layer down to the raw input tokens.
'''
def trace_attn_viz_func(tf_attns, ctx_attns, sentence, seq_id, save_fig=False,
                        context_viz_include=False, linear_layer_include=True,
                        reduce="none", view="small"):
    # input params
    tokens = tf_attns.shape[2]
    layers = 6
    heads = 8
    if view == "small":
        point_offset = 0.1
        box_offset = 0.3
    elif view == "normal":
        point_offset = 0.3
        box_offset = 0.3

    # generate points for making the connection plot
    layer_offset = point_offset * (tokens-1)
    total_width = layer_offset * heads + (box_offset*(heads - 1))
    center = total_width * 0.5
    token_offset = 2.5
    token_pos_x = [0.0] * tokens
    for i in range(tokens):
        token_pos_x[i] = i*token_offset
    center_text = token_pos_x[-1] * 0.5
    for i in range(tokens):
        token_pos_x[i] = token_pos_x[i] + (center - center_text) # add offset

    # accumulative attention calculation
    head_level_attentions = []
    for h in range(heads):
        tf_attn = tf_attns[h]
        level_attentions = []
        pre_attn = ctx_attns.clone()
        level_attentions.append(ctx_attns.clone())
        for i in reversed(range(layers)):
            curr_tf_attn = torch.matmul(pre_attn, tf_attn[i])
            level_attentions.append(curr_tf_attn.clone())
            pre_attn = curr_tf_attn
        level_attentions = torch.stack(level_attentions, dim=0)
        head_level_attentions.append(level_attentions.clone())
    head_level_attentions = torch.stack(head_level_attentions, dim=0)
    token_attn_accum = head_level_attentions[:,-1,:].sum(dim=0)
    print("Raw attention score:")
    print(token_attn_accum)
    if reduce == "softmax":
        token_attn_accum = F.softmax(token_attn_accum, dim=-1)
    elif reduce == "min_max":
        max_attn = token_attn_accum.max()
        token_attn_accum = token_attn_accum * 1. / max_attn
    elif reduce == "none":
        token_attn_accum = token_attn_accum.clamp(0, 1)

    head_level_attentions_repeat = torch.stack([head_level_attentions]*tokens, dim=-1)
               
    # generate pair (of two points) and attention weight of the pair
    pair_weight = dict()
    pair_layer = dict()
    pair_head = dict()
    
    # the token layer and ctx layer connection points
    token_xys_top = []
    token_xys_bottom = []
    token_xys_top = []
    ctx_xys_bottom = []

    # create figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) # nrows, ncols, index
    
    for i in range(heads):
        for j in range(layers):
            # for layer j, head i, we want to all the point pairs and weight
            base_x = i*(layer_offset + box_offset)
            if linear_layer_include:
                base_y = (j+1)*(layer_offset + box_offset)
            else:
                base_y = (j+1)*(layer_offset)
            # iteration through all the tokens to form pairs
            for k in range(tokens):
                k_x = base_x + k*(point_offset)
                k_y = base_y
                if j == 0:
                    # handle the input lower token
                    token_y = j*(layer_offset) + box_offset
                    token_x = token_pos_x[k]
                    # if first time, we annotate the tokens with actual string
                    if i == 0:
                        ax.annotate(sentence[k],
                                    (token_x, token_y),
                                    textcoords="offset points",
                                    xytext=(0,-15),
                                    ha='center',
                                    size=14,
                                    alpha=token_attn_accum[k])
                    pair_weight[((k_x, k_y),(token_x, token_y))] = \
                        head_level_attentions_repeat[i, layers - j, k, 0] 
                    pair_layer[((k_x, k_y),(token_x, token_y))] = j
                    pair_head[((k_x, k_y),(token_x, token_y))] = i
                else:
                    if linear_layer_include:
                        # handle the linear linears
                        last_x = k_x
                        last_y = k_y - box_offset
                        pair_weight[((k_x, k_y),(last_x, last_y))] = \
                            head_level_attentions_repeat[i, layers - j, k, 0] 
                        pair_layer[((k_x, k_y),(last_x, last_y))] = j
                        pair_head[((k_x, k_y),(last_x, last_y))] = i
                for l in range(tokens):
                    l_x = base_x + l*(point_offset)
                    l_y = base_y + layer_offset
                    # handle the upper ctx
                    if context_viz_include:
                        if j == layers - 1 and k == 0: # make sure only connect once!
                            ctx_y = l_y + 1.0*(layer_offset)
                            ctx_x = base_x + point_offset*tokens*0.5
                            pair_weight[((l_x, l_y),(ctx_x, ctx_y))] = \
                                ctx_attns[l]
                            pair_layer[((l_x, l_y),(ctx_x, ctx_y))] = j
                            pair_head[((l_x, l_y),(ctx_x, ctx_y))] = i
                    pair_weight[((k_x, k_y),(l_x, l_y))] = \
                        tf_attns[i, j, l, k] * \
                        head_level_attentions_repeat[i, layers - j - 1, l, k]
                    pair_layer[((k_x, k_y),(l_x, l_y))] = j
                    pair_head[((k_x, k_y),(l_x, l_y))] = i

    # colors = ['red', 'chocolate', 'gold', 'greenyellow', 'darkgreen', 'aqua', 'blue', 'fuchsia']
    colors = ['black'] * 8
    # plot all the points, and connect them with lines
    for pair in pair_weight.keys():
        plt.plot([pair[0][0], pair[1][0]], 
                [pair[0][1], pair[1][1]], 
                'w-',
                linewidth=2.5,
                alpha=pair_weight[pair],
                color=colors[pair_head[pair]])
    ax.set_facecolor('white')
    if linear_layer_include:
        plt.xticks([(i*(layer_offset + box_offset)+layer_offset*0.5) for i in range(heads)],
                   ['0', '1', '2', '3', '4', '5', '6', '7'], fontsize=15)
        plt.yticks([(i*(layer_offset + box_offset)+layer_offset*0.5) for i in range(layers+1)],
                   ['input', '0', '1', '2', '3', '4', '5'], fontsize=15)
    else:
        plt.xticks([(i*(layer_offset + box_offset)+layer_offset*0.5) for i in range(heads)],
                   ['0', '1', '2', '3', '4', '5', '6', '7'], fontsize=15)
        plt.yticks([(i*(layer_offset)+layer_offset*0.5) for i in range(layers+1)],
                   ['input', '0', '1', '2', '3', '4', '5'], fontsize=15)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.xlabel('Heads', fontsize=25)
    plt.ylabel('Layers', fontsize=25)
    ax.xaxis.set_label_position('top')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    # the output plot can be rescaled using this
    plt.show()

    if save_fig:
        # resize to make sure it is clear
        plt.savefig('../tf_attns_plots/' + str(seq_id) + '.png', bbox_inches='tight',dpi=100)
        plt.close(fig)