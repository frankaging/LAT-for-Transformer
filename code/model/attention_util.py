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
    plt.rcParams["font.family"] = "Times New Roman"
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
                        context_viz_include=False, linear_layer_include=False,
                        reduce="none", view="small"):
    # input params
    tokens = tf_attns.shape[2]
    layers = 6
    heads = 8
    point_offset = 0.3
    box_offset = 0.3
    if view == "small":
        point_offset = 0.1
        box_offset = 0.3
    elif view == "normal":
        point_offset = 0.3
        box_offset = 0.3
    elif view == "customize":
        point_offset = 3.5
        box_offset = 5.2

    # generate points for making the connection plot
    layer_offset = point_offset * (tokens-1)
    total_width = layer_offset * heads + (box_offset*(heads - 1))
    center = total_width * 0.5
    token_offset = 36
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
    fig = plt.figure(figsize=(20,10))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
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
                                    xytext=(0,-30),
                                    ha='center',
                                    size=36,
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
                        (tf_attns[i, j, l, k] * \
                        head_level_attentions_repeat[i, layers - j - 1, l, k])*0.997 + 0.003
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
                color=colors[pair_head[pair]],
                marker='.',
                markersize=14)
    ax.set_facecolor('white')
    if linear_layer_include:
        plt.xticks([(i*(layer_offset + box_offset)+layer_offset*0.5) for i in range(heads)],
                   ['1', '2', '3', '4', '5', '6', '7', '8'], fontsize=38)
        plt.yticks([(i*(layer_offset + box_offset)+layer_offset*0.5) for i in range(layers+1)],
                   ['input', '1', '2', '3', '4', '5', '6'], fontsize=38)
    else:
        plt.xticks([(i*(layer_offset + box_offset)+layer_offset*0.5) for i in range(heads)],
                   ['1', '2', '3', '4', '5', '6', '7', '8'], fontsize=38)
        plt.yticks([(i*(layer_offset)+layer_offset*0.5) for i in range(layers+1)],
                   ['input', '1', '2', '3', '4', '5', '6'], fontsize=38)
    plt.ylim(bottom=-15)
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.xlabel('Heads', fontsize=38, labelpad=10)
    plt.ylabel('Layers', fontsize=38, labelpad=-60)
    ax.xaxis.set_label_position('top')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    # the output plot can be rescaled using this
    if save_fig:
        plt.savefig('../tf_attns_plots/' + str(seq_id) + '.png', bbox_inches='tight',dpi=100)
        plt.close(fig)
    else:
        plt.show()


'''
Visualization function for attention distribution on targeted words in each head.
'''
def head_heatmap_viz_func(id_tf_attns, id_ctx_attns, sentece_token_rate, length_filter=0,
                          include_words=[1,5], save_fig=False, max_cbar=0.25,
                          dataset="sst", cmap="Blues"):
    
    include_ids = set([])
    internal_include = [1,5]
    for _id in id_tf_attns.keys():
        if 1 in sentece_token_rate[_id] or \
            5 in sentece_token_rate[_id]:
            include_ids.add(_id)
    print("Words Considered: ")
    print(include_words)
    print("Test Set Size: ")
    print(len(id_tf_attns.keys()))
    # add a filter about length
    id_selected = set([])
    for _id in include_ids:
        if len(sentece_token_rate[_id]) >= length_filter:
            id_selected.add(_id)
    print("Selected Set Size: ")
    print(len(id_selected))

    # random percentage is different for different tags
    random_pc_all = random_percentage(sentece_token_rate, id_selected, include_words=[1,5])
    random_pc_neg = random_percentage(sentece_token_rate, id_selected, include_words=[1])
    random_pc_pos = random_percentage(sentece_token_rate, id_selected, include_words=[5])
    print(random_pc_all, random_pc_neg, random_pc_pos)
    
    layers = 6
    heads = 8
    emo_map = np.zeros((3, heads))
    for seq in id_selected:
        tf_attns = torch.FloatTensor(id_tf_attns[seq])
        ctx_attns = torch.FloatTensor(id_ctx_attns[seq])
        
        pre_attn = ctx_attns.clone()
        # load token level SST label
        token_rate = sentece_token_rate[seq]
        # input params
        tokens = tf_attns.shape[2]

        # for each head
        head_level_attentions = []
        for i in range(heads):
            tf_attn = tf_attns[i]
            pre_attn = ctx_attns.clone()
            for j in reversed(range(layers)):
                curr_tf_attn = torch.matmul(pre_attn, tf_attn[j])
                pre_attn = curr_tf_attn.clone()
            current_attns = pre_attn.clone() # WARNING: You cannot do a softmax again here to normalize!
            
            # for this head, till the first layer, what does it focus on?
            attn_sum_all = 0.0
            for t in range(tokens):
                
                # we add if the token is very emotionally expressed
                if token_rate[t] in [1,5]:
                    attn_sum_all += current_attns[t]
            attn_sum_all = attn_sum_all # minus the random prob
            emo_map[0, i] += attn_sum_all - random_pc_all

            attn_sum_pos = 0.0
            for t in range(tokens):
                # we add if the token is very emotionally expressed
                if token_rate[t] in [5]:
                    attn_sum_pos += current_attns[t]
            attn_sum_pos = attn_sum_pos # minus the random prob
            emo_map[1, i] += attn_sum_pos - random_pc_pos
            
            attn_sum_neg = 0.0
            for t in range(tokens):
                # we add if the token is very emotionally expressed
                if token_rate[t] in [1]:
                    attn_sum_neg += current_attns[t]
            attn_sum_neg = attn_sum_neg # minus the random prob
            emo_map[2, i] += attn_sum_neg - random_pc_neg

    emo_map = emo_map/len(id_selected)
    
    # generate customerized labels
    emo_map_str = []
    for i in range(emo_map.shape[0]):
        inner_str = [] 
        for j in range(emo_map.shape[1]):
            if emo_map[i, j] > 0:
                fmt_str = '{:.3f}'.format(emo_map[i, j])
                inner_str.append(fmt_str[1:])
            else:
                fmt_str = '{:.3f}'.format(emo_map[i, j])
                inner_str.append("-"+fmt_str[2:])
        emo_map_str.append(inner_str)
    emo_map_str = np.array(emo_map_str)
    
    # heatmap viz
    emo_map = emo_map
    import seaborn as sns
    fig = plt.figure(figsize=(14,10))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    ax = sns.heatmap(emo_map, annot=emo_map_str, fmt = '', cmap=cmap,
                     linewidths=2,
                     annot_kws={"fontsize":35},
                     vmin=0,
                     cbar_kws=dict(shrink=0.4, aspect=5))
    ax.set_aspect("equal")
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=35)
    from matplotlib.ticker import StrMethodFormatter
    from matplotlib.ticker import FuncFormatter
    def my_formatter(x, pos):
        """Format 1 as 1, 0 as 0, and all values whose absolute values is between
        0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
        formatted as -.4)."""
        val_str = '{:.3f}'.format(x)
        if np.abs(x) >= 0 and np.abs(x) < 1:
            return val_str.replace("0", "", 1)
        else:
            return val_str
    major_formatter = FuncFormatter(my_formatter)
    cbar.ax.yaxis.set_major_formatter(major_formatter)

    ax.xaxis.tick_top()
    plt.yticks([0.4,1.3,2.3], ('all','pos','neg'), size = 40)
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8], size = 40)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.xlabel('Heads', fontsize=45)
    ax.xaxis.set_label_position('top')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    fig_title = dataset.upper()

    ax.annotate(fig_title,  # Your string
                # The point that we'll place the text in relation to 
                xy=(0.5, 0.28), 
                # Interpret the x as axes coords, and the y as figure coords
                xycoords=('axes fraction', 'figure fraction'),
                # The distance from the point that the text will be at
                xytext=(0, 0),  
                # Interpret `xytext` as an offset in points...
                textcoords='offset points',
                # Any other text parameters we'd like
                size=40, ha='center', va='bottom',
                color='black')

    if save_fig:
        name = dataset + "_"
        if len(include_words) == 2:
            name += "head_all.png"
        else:
            if include_words[0] == 1:
                name += "head_neg.png"
            else:
                name += "head_pos.png"
        plt.savefig('../tf_attns_plots/' + name, bbox_inches='tight',dpi=100)
        plt.close(fig)
    else:
        plt.show()



'''
Deprecated:

Visualization function for attention distribution on targeted words in each head.
'''
def head_heatmap_viz_func_individual(id_tf_attns, sentece_token_rate, length_filter=19,
                          include_words=[1,5], save_fig=False, max_cbar=0.25,
                          dataset="sst"):

    include_ids = set([])
    internal_include = [1,5]
    for _id in id_tf_attns.keys():
        if 1 in sentece_token_rate[_id] or \
            5 in sentece_token_rate[_id]:
            include_ids.add(_id)
    print("Words Considered: ")
    print(include_words)
    print("Test Set Size: ")
    print(len(id_tf_attns.keys()))
    # add a filter about length
    id_selected = set([])
    for _id in include_ids:
        if len(sentece_token_rate[_id]) >= length_filter:
            id_selected.add(_id)
    print("Selected Set Size: ")
    print(len(id_selected))

    layers = 6
    heads = 8
    emo_map = np.zeros((layers, heads))
    for seq in id_selected:
        tf_attns = torch.FloatTensor(id_tf_attns[seq])
        # load token level SST label
        token_rate = sentece_token_rate[seq]
        # input params
        tokens = tf_attns.shape[2]
        # for each head
        for i in range(heads):
            for j in reversed(range(layers)):
                unit_tf_attns = tf_attns[i,j] # reverse the layer here
                tf_attns_sum = unit_tf_attns.sum(dim=0).unsqueeze(dim=-1)
                current_attns = None
                for k in reversed(range(j)):
                    # if it is the last layer
                    if k == j - 1:
                        current_attns = tf_attns_sum * tf_attns[i,k]
                    else:
                        current_attns = torch.matmul(current_attns, tf_attns[i,k])
                # extract related attentions paid to meaningful tokens
                if current_attns is not None:
                    current_attns = current_attns.sum(dim=0)
                else:
                    current_attns = tf_attns_sum.squeeze(dim=-1)
                attn_sum = 0.0
                token_acount = 0
                current_attns = F.softmax(current_attns) # reduce function using softmax
                for t in range(tokens):
                    # we add if the token is very emotionally expressed
                    if token_rate[t] in include_words:
                        attn_sum += current_attns[t]
                attn_sum = attn_sum*1.0
                emo_map[layers - j - 1, i] += attn_sum # probably let us reverse the y axis here ?

    emo_map = emo_map/len(id_selected)

    # heatmap viz
    emo_map = emo_map
    import seaborn as sns
    fig = plt.figure(figsize=(14,10))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    ax = sns.heatmap(emo_map, annot=True, fmt=".3f", cmap="Greys",
                     xticklabels=[1, 2, 3, 4, 5, 6, 7, 8],
                     yticklabels=[6, 5, 4, 3, 2, 1], linewidths=2,
                     annot_kws={"fontsize":33},
                     vmin=0,
                     cbar_kws=dict(shrink=0.5, aspect=5))
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=35)
    from matplotlib.ticker import StrMethodFormatter
    cbar.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
    ax.xaxis.tick_top()
    ax.set_yticklabels([6, 5, 4, 3, 2, 1], size = 45)
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8], size = 45)
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.xlabel('Heads', fontsize=45)
    plt.ylabel('Layers', fontsize=45)
    ax.xaxis.set_label_position('top')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    fig_title = ""
    if len(include_words) == 2:
        fig_title = "Very Positive and Very Negative Words"
    else:
        if include_words[0] == 1:
            fig_title = "Very Negative Words"
        else:
            fig_title = "Very Positive Words"

    ax.annotate(fig_title,  # Your string
                # The point that we'll place the text in relation to 
                xy=(0.5, 0.03), 
                # Interpret the x as axes coords, and the y as figure coords
                xycoords=('axes fraction', 'figure fraction'),
                # The distance from the point that the text will be at
                xytext=(0, 0),  
                # Interpret `xytext` as an offset in points...
                textcoords='offset points',
                # Any other text parameters we'd like
                size=35, ha='center', va='bottom',
                color='black')

    if save_fig:
        name = dataset + "_"
        if len(include_words) == 2:
            name += "head_all.png"
        else:
            if include_words[0] == 1:
                name += "head_neg.png"
            else:
                name += "head_pos.png"
        plt.savefig('../tf_attns_plots/' + name, bbox_inches='tight',dpi=100)
        plt.close(fig)
    else:
        plt.show()

    percentage_sum = 0.0
    for seq in id_selected:
        # load token level SST label
        token_rate = sentece_token_rate[seq]
        # input params
        tokens = len(token_rate)
        count = 0
        for t in range(tokens):
            if token_rate[t] in include_words:
                count += 1
        percentage = count*1.0 / tokens
        percentage_sum += percentage
    print("Average percentage with randomly assigned attention: ")
    print(percentage_sum*1.0/len(id_selected))