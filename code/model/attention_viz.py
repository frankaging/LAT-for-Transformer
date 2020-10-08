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
from attention_util import *

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ])
logger = logging.getLogger()

def eval_ccc(y_true, y_pred):
    """Computes concordance correlation coefficient."""
    true_mean = np.mean(y_true)
    true_var = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_var = np.var(y_pred)
    covar = np.cov(y_true, y_pred, bias=True)[0][1]
    ccc = 2*covar / (true_var + pred_var +  (pred_mean-true_mean) ** 2)
    return ccc

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

'''
helper to chunknize the data for each a modality
'''
def generateInputChunkHelper(data_chunk, length_chunk, tensor=True):
    # sort the data with length from long to short
    combined_data = list(zip(data_chunk, length_chunk))
    combined_data.sort(key=itemgetter(1),reverse=True)
    data_sort = []
    for pair in combined_data:
        data_sort.append(pair[0])

    if tensor:
        # produce the operatable tensors
        data_sort_t = torch.tensor(data_sort, dtype=torch.float)
        return data_sort_t
    else:
        return data_sort

'''
yielding training batch for the training process
'''
def generateTrainBatch(input_data, input_target, input_length, token_lengths, args, batch_size=25):
    # TODO: support input_data as a dictionary
    # get chunk
    input_size = len(input_data[list(input_data.keys())[0]]) # all values have same size
    index = [i for i in range(0, input_size)]
    if batch_size != 1:
        shuffle(index)
    shuffle_chunks = [i for i in chunks(index, batch_size)]
    for chunk in shuffle_chunks:
        # chunk yielding data
        yield_input_data = {}
        # same across a single chunk
        target_chunk = [input_target[index] for index in chunk]
        length_chunk = [input_length[index] for index in chunk]
        token_length_chunk = [token_lengths[index] for index in chunk]
        # max length
        max_length = max(length_chunk)
        # max token length
        max_token_length = max([max(chunk) for chunk in token_length_chunk])
        # mod data generating
        for mod in list(input_data.keys()):
            data_chunk = [input_data[mod][index] for index in chunk]
            data_chunk_sorted = \
                generateInputChunkHelper(data_chunk, length_chunk)
            data_chunk_sorted = data_chunk_sorted[:,:max_length,:max_token_length,:]
            yield_input_data[mod] = data_chunk_sorted
        # target generating
        target_sort = \
            generateInputChunkHelper(target_chunk, length_chunk)
        target_sort = target_sort[:,:max_length]
        # token length generating
        token_length_sort = \
            generateInputChunkHelper(token_length_chunk, length_chunk, tensor=False)

        # mask generation for the whole batch
        lstm_masks = torch.zeros(target_sort.size()[0], target_sort.size()[1], 1, dtype=torch.float)
        length_chunk.sort(reverse=True)
        for i in range(lstm_masks.size()[0]):
            lstm_masks[i,:length_chunk[i]] = 1

        # yielding for each batch
        yield (yield_input_data, torch.unsqueeze(target_sort, dim=2), lstm_masks, length_chunk, token_length_sort)

def evaluateOnEval(input_data, input_target, lengths, token_lengths, model, criterion, args, fig_path=None):
    model.eval()
    predictions = []
    weights_total = []
    actuals = []
    data_num = 0
    loss, ccc = 0.0, []
    count = 0
    index = 0
    total_vid_count = len(input_data[list(input_data.keys())[0]])
    for (data, target, mask, lengths, token_lengths) in generateTrainBatch(input_data,
                                                            input_target,
                                                            lengths,
                                                            token_lengths, 
                                                            args,
                                                            batch_size=1):
        print("Video #: " + str(index+1) + "/" + str(total_vid_count))
        # send to device
        mask = mask.to(args.device)
        # send all data to the device
        for mod in list(data.keys()):
            data[mod] = data[mod].to(args.device)
        target = target.to(args.device)
        # Run forward pass
        output = model.forward(data, lengths, token_lengths, mask)
        # Also get the weight
        weights = model.backward_nlap(data, lengths, token_lengths, mask)
        weights_total.append(weights)

        predictions.append(output.reshape(-1).tolist())
        actuals.append(target.reshape(-1).tolist())
        # Compute loss
        loss += criterion(output, target)
        # Keep track of total number of time-points
        data_num += sum(lengths)
        # Compute correlation and CCC of predictions against ratings
        output = torch.squeeze(torch.squeeze(output, dim=2), dim=0).cpu().detach().numpy()
        target = torch.squeeze(torch.squeeze(target, dim=2), dim=0).cpu().detach().numpy()
        if count == 0:
            # print(output)
            # print(target)
            count += 1
        curr_ccc = eval_ccc(output, target)
        ccc.append(curr_ccc)
        index += 1
    # Average losses and print
    loss /= data_num
    return ccc, predictions, actuals, weights_total

def plot_predictions(dataset, predictions, metric, args, fig_path=None):
    """Plots predictions against ratings for representative fits."""
    # Select top 4 and bottom 4
    sel_idx = np.concatenate((np.argsort(metric)[-4:][::-1],
                              np.argsort(metric)[:4]))
    sel_metric = [metric[i] for i in sel_idx]
    sel_true = [dataset.orig['ratings'][i] for i in sel_idx]
    sel_pred = [predictions[i] for i in sel_idx]
    for i, (true, pred, m) in enumerate(zip(sel_true, sel_pred, sel_metric)):
        j, i = (i // 4), (i % 4)
        args.axes[i,j].cla()
        args.axes[i,j].plot(true, 'b-')
        args.axes[i,j].plot(pred, 'c-')
        args.axes[i,j].set_xlim(0, len(true))
        args.axes[i,j].set_ylim(-1, 1)
        args.axes[i,j].set_title("Fit = {:0.3f}".format(m))
    plt.tight_layout()
    plt.draw()
    if fig_path is not None:
        plt.savefig(fig_path)
    plt.pause(1.0 if args.test else 0.001)

def plot_eval(pred_sort, ccc_sort, actual_sort, window_size=1):
    sub_graph_count = len(pred_sort)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(1, 7):
        ax = fig.add_subplot(2, 3, i)

        ccc = ccc_sort[i-1]
        pred = pred_sort[i-1]
        actual = actual_sort[i-1]
        minL = min(len(pred), len(actual))
        pred = pred[:minL]
        actual = actual[:minL]
        t = []
        curr_t = 0.0
        for i in pred:
            t.append(curr_t)
            curr_t += window_size
        pred_line, = ax.plot(t, pred, '-' , color='r', linewidth=2.0, label='Prediction')
        ax.legend()
        actual_line, = ax.plot(t, actual, '-', color='b', linewidth=2.0, label='True')
        ax.legend()
        ax.set_ylabel('valence(0-10)')
        ax.set_xlabel('time(s)')
        ax.set_title('ccc='+str(ccc)[:5])
    plt.show()
    # plt.savefig("./lstm_save/top_ccc.png")

def save_predictions(dataset, predictions, path):
    for p, seq_id in zip(predictions, dataset.seq_ids):
        df = pd.DataFrame(p, columns=['rating'])
        fname = "target_{}_{}_normal.csv".format(*seq_id)
        df.to_csv(os.path.join(path, fname), index=False)

def save_params(args, model, train_stats, test_stats):
    fname = 'param_hist.tsv'
    df = pd.DataFrame([vars(args)], columns=vars(args).keys())
    df = df[['modalities', 'batch_size', 'split', 'epochs', 'lr',
             'sup_ratio', 'base_rate']]
    for k in ['ccc_std', 'ccc']:
        v = train_stats.get(k, float('nan'))
        df.insert(0, 'train_' + k, v)
    for k in ['ccc_std', 'ccc']:
        v = test_stats.get(k, float('nan'))
        df.insert(0, 'test_' + k, v)
    df.insert(0, 'model', [model.__class__.__name__])
    df['embed_dim'] = model.embed_dim
    df['h_dim'] = model.h_dim
    df['attn_len'] = model.attn_len
    df['ar_order'] = [float('nan')]
    df.set_index('model')
    df.to_csv(fname, mode='a', header=(not os.path.exists(fname)), sep='\t')

def save_checkpoint(modalities, mod_dimension, window_size, model, path):
    checkpoint = {'modalities': modalities, 'mod_dimension' : mod_dimension, 'window_size' : window_size, 'model': model.state_dict()}
    torch.save(checkpoint, path)

def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    return checkpoint

def load_data(modalities, data_dir, eval_dir=None):
    print("Loading data...")
    if eval_dir == None:
        train_data = load_dataset(modalities, data_dir, 'Train',
                                truncate=True, item_as_dict=True)
        test_data = load_dataset(modalities, data_dir, 'Valid',
                                truncate=True, item_as_dict=True)
        print("Done.")
        return train_data, test_data
    eval_data = load_dataset(modalities, data_dir, eval_dir,
                             truncate=True, item_as_dict=True)
    print("Loading Eval Set Done.")
    return eval_data

def textInputHelper(input_data, window_size, channel):
    # channel features
    vectors_raw = input_data[channel]
    ts = input_data["linguistic_timer"]

    #  get the window size and repeat rate if oversample is needed
    oversample = int(window_size[channel]/window_size['ratings'])
    window_size = window_size[channel]

    video_vs = []
    count_v = 0
    current_time = 0.0
    window_vs = []
    while count_v < len(vectors_raw):
        t = ts[count_v]
        if type(t) == list:
            t = t[0]
        if t <= current_time + window_size:
            window_vs.append(vectors_raw[count_v])
            count_v += 1
        else:
            if len(window_vs) == 0:
                window_vs.append('null')
            for i in range(0, oversample):
                video_vs.append(window_vs)
            window_vs = []
            current_time += window_size
    return video_vs

def videoInputHelper(input_data, window_size, channel):
    # channel features
    vectors_raw = input_data[channel]
    ts = input_data["linguistic_timer"]
    # remove nan values
    vectors = []
    for vec in vectors_raw:
        inner_vec = []
        for v in vec:
            if np.isnan(v):
                inner_vec.append(0)
            else:
                inner_vec.append(v)
        vectors.append(inner_vec)

    #  get the window size and repeat rate if oversample is needed
    oversample = int(window_size[channel]/window_size['ratings'])
    window_size = window_size[channel]
    mod_dimension = {'linguistic' : 300}

    video_vs = []
    count_v = 0
    current_time = 0.0
    window_vs = []
    while count_v < len(vectors):
        t = ts[count_v]
        if type(t) == list:
            t = t[0]
        if t <= current_time + window_size:
            window_vs.append(vectors[count_v])
            count_v += 1
        else:
            if len(window_vs) == 0:
                if count_v > 0 :
                    # append last one again
                    pad_vec = np.copy(vectors[count_v-1])
                    window_vs.append(pad_vec)
                else:
                    # or enforce append first element
                    if len(vectors) != 0:
                        pad_vec = np.copy(vectors[0])
                        window_vs.append(pad_vec)
                    else:
                        # or simply 0 like will never happened!
                        pad_vec = [0.0] * mod_dimension[channel]
                        pad_vec = np.array(pad_vec)
                        window_vs.append(pad_vec)
            for i in range(0, oversample):
                temp = np.array(window_vs)
                video_vs.append(temp)
            window_vs = []
            current_time += window_size
    # TODO: we are only taking average from each window for image
    #if channel == 'image':
    # data = np.asarray(video_vs)
    # data = np.average(data, axis=1)
    # video_vs = np.expand_dims(data, axis=1).tolist()
    return video_vs

def ratingInputHelper(input_data, window_size):
    ratings = input_data['ratings']
    ts = input_data['ratings_timer']
    window_size = window_size['ratings']

    current_time = 0.0
    count_r = 0
    window_rs = []
    video_rs = []
    while count_r < len(ratings):
        t = ts[count_r]
        if t <= current_time + window_size:
            window_rs.append(ratings[count_r])
            count_r += 1
        else:
            avg_r = sum(window_rs)*1.0/len(window_rs)
            video_rs.append(avg_r)
            window_rs = []
            current_time += window_size
    return video_rs

'''
Construct inputs for different channels: emotient, linguistic, ratings, etc..
'''
def constructInput(input_data, window_size, channels):
    ret_input_features = {}
    ret_ratings = []
    for data in input_data:
        # print(data['linguistic_timer'])
        # channel features
        minL = 99999999
        for channel in channels:
            if channel != "linguistic_text":
                video_vs = videoInputHelper(data, window_size, channel)
            else:
                video_vs = textInputHelper(data, window_size, channel)
            if channel not in ret_input_features.keys():
                ret_input_features[channel] = []
            ret_input_features[channel].append(video_vs)
            if len(video_vs) < minL:
                minL = len(video_vs)
        video_rs = ratingInputHelper(data, window_size)
        # print("video_rs vector size: " + str(len(video_rs)))
        if len(video_rs) < minL:
            minL = len(video_rs)
        # concate
        for channel in channels:
             ret_input_features[channel][-1] = ret_input_features[channel][-1][:minL]
        ret_ratings.append(video_rs[:minL])
    return ret_input_features, ret_ratings

def padInputHelper(input_data, dim, old_version=False):
    output = []
    max_num_vec_in_window = 0
    max_num_windows = 0
    seq_lens = []
    token_lens = []
    for data in input_data:
        if max_num_windows < len(data):
            max_num_windows = len(data)
        seq_lens.append(len(data))
        if max_num_vec_in_window < max([len(w) for w in data]):
            max_num_vec_in_window = max([len(w) for w in data])
        token_lens.append([len(w) for w in data])

    padVec = [0.0]*dim
    for vid in input_data:
        vidNewTmp = []
        for wind in vid:
            windNew = [padVec] * max_num_vec_in_window
            windNew[:len(wind)] = wind
            vidNewTmp.append(windNew)
        vidNew = [[padVec] * max_num_vec_in_window]*max_num_windows
        vidNew[:len(vidNewTmp)] = vidNewTmp
        output.append(vidNew)
    return output, seq_lens, token_lens

'''
pad every sequence to max length, also we will be padding windows as well
'''
def padInput(input_data, channels, dimensions):
    # input_features <- list of dict: {channel_1: [117*features],...}
    ret = {}
    seq_lens = []
    for channel in channels:
        pad_channel, seq_lens, token_lens = padInputHelper(input_data[channel], dimensions[channel])
        ret[channel] = pad_channel
    return ret, seq_lens, token_lens
def getSeqList(seq_ids):
    ret = []
    for seq_id in seq_ids:
        ret.append(seq_id[0]+"_"+seq_id[1])
    return ret
'''
pad targets
'''
def padRating(input_data, max_len):
    output = []
    # pad ratings
    for rating in input_data:
        ratingNew = [0]*max_len
        ratingNew[:len(rating)] = rating
        output.append(ratingNew)
    return output

def softmax(x_in, axis=None):
    x = np.array(x_in)
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def normalize_w(x):
    if max(x) == min(x):
        return [sum(x)*1.0/len(x) for w in x]
    else:
        return [(w - min(x))*1.0/(max(x) - min(x)) for w in x]

def normalize_lap(x):
    abs_w = [abs(w) for w in x]
    return [w*1.0/max(abs_w) for w in x]

def generate_class(raw_in):
    _class = dict()
    for seq in raw_in.keys():
        rate = raw_in[seq]
        if rate >= 0.0 and rate <= 0.2:
            _class[seq] = 1
        elif rate > 0.2 and rate <= 0.4:
            _class[seq] = 2
        elif rate > 0.4 and rate <= 0.6:
            _class[seq] = 3
        elif rate > 0.6 and rate <= 0.8:
            _class[seq] = 4
        elif rate > 0.8 and rate <= 1.0:
            _class[seq] = 5
        else:
            assert(False)
    return _class

def padFeaturesSST(features):
    # get max length
    seq_len = [len(seq) for seq in features]
    seq_len_max = max(seq_len)
    pad_vec = [0.0]*300
    padded_feature = []
    for seq in features:
        temp_feature = [pad_vec] * seq_len_max
        temp_feature[:len(seq)] = seq
        padded_feature.append(temp_feature)
    return padded_feature, seq_len
        
def sortSST(data_chunk, length_chunk, tensor=True):
    # sort the data with length from long to short
    combined_data = list(zip(data_chunk, length_chunk))
    combined_data.sort(key=itemgetter(1),reverse=True)
    data_sort = []
    for pair in combined_data:
        data_sort.append(pair[0])
    if tensor:
        # produce the operatable tensors
        data_sort_t = torch.tensor(data_sort, dtype=torch.float)
        return data_sort_t
    return data_sort

def generateBatchSST(input_data, input_target, seq_ids, args, batch_size=1):
    # select batch sentence id
    index = [i for i in range(0, len(seq_ids))]
    shuffle_chunks = [i for i in chunks(index, batch_size)] # contains array index
    for chunk in shuffle_chunks:
        chunk_ids = [seq_ids[index] for index in chunk]
        # sort feature
        features = [input_data[_id] for _id in chunk_ids]
        padded_feature, seq_len = padFeaturesSST(features)
        sort_feature = sortSST(padded_feature, seq_len)
        # sort target
        targets = [input_target[_id] for _id in chunk_ids]
        sort_targets = sortSST(targets, seq_len)
        # sorted chunk id
        sort_chunk_ids = sortSST(chunk_ids, seq_len, tensor=False)
        # sort length
        seq_len.sort(reverse=True)
        # mask
        batch_size = len(seq_len)
        max_len = max(seq_len)
        mask = torch.zeros(batch_size, max_len, dtype=torch.float)
        for i in range(mask.size()[0]):
            mask[i,:seq_len[i]] = 1
        yield sort_feature, sort_targets, seq_len, mask, sort_chunk_ids

def oneHotVector(sort_targets):
    n_class = 5
    batch_size = sort_targets.shape[0]
    target = torch.zeros((batch_size, int(n_class)), dtype=torch.float).to(args.device)
    for i in range(batch_size):
        target[i][int(sort_targets[i]) - 1] = 1.0
    return target

def calculate_accuracy(predict, actual):
    _, predicted = torch.max(predict, dim=-1)
    multiclass = 0
    for i in range(predict.shape[0]):
        if int(predicted[i]) == int(actual[i]) - 1:
            multiclass += 1

    binary = 0
    binary_total = 0 # will not count nuetural cases
    for i in range(predict.shape[0]):
        if int(actual[i]) != 3:
            binary_total += 1
            if int(actual[i]) < 3 and int(predicted[i]) < 2:
                binary += 1
            elif int(actual[i]) > 3 and int(predicted[i]) > 2:
                binary += 1

    return multiclass, binary, binary_total

def stringOut(sort_targets, output):
    _, predicted = torch.max(output, dim=-1)
    ret = []
    for i in range(sort_targets.shape[0]):
        actual = ''
        if int(sort_targets[i]) == 5:
            actual = "$++$"
        elif int(sort_targets[i]) == 4:
            actual = "$+$"
        elif int(sort_targets[i]) == 3:
            actual = "$.$"
        elif int(sort_targets[i]) == 2:
            actual = "$-$"
        elif int(sort_targets[i]) == 1:
            actual = "$--$"
        predict = ''
        if int(predicted[i]) == 4:
            predict = "$++$"
        elif int(predicted[i]) == 3:
            predict = "$+$"
        elif int(predicted[i]) == 2:
            predict = "$.$"
        elif int(predicted[i]) == 1:
            predict = "$-$"
        elif int(predicted[i]) == 0:
            predict = "$--$"
        ret.append((actual, predict))
    return ret

def get_attn_weight(i,j,k,l):
    return random.uniform(0, 1)

def extract_attn_weight(args):
    print("Start analyzing SST trained models ...")
    # Fix random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    # clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Convert device string to torch.device
    args.device = (torch.device(args.device) if torch.cuda.is_available()
                   else torch.device('cpu'))

    # Loading the data
    print("Loading SST data ...")
    data_folder = args.data_dir
    test_data = pickle.load( open( data_folder + "id_embed_test.p", "rb" ) )
    test_target = pickle.load( open( data_folder + "id_rating_test.p", "rb" ) )
    assert(len(test_data) == len(test_target))
    print("Verified SST data ...")
    print("Test Set Size: ", len(test_data))

    # set the prediction categories
    test_class = generate_class(test_target)

    args.modalities = ['linguistic']
    mod_dimension = {'linguistic' : 300}
    # loss function define
    criterion = nn.BCELoss(reduction='sum')
    # construct model
    model = TransformerLinearAttn(mods=args.modalities, dims=mod_dimension, device=args.device)
    # load model
    model_path = args.model_path
    checkpoint = load_checkpoint(model_path, args.device)
    model.load_state_dict(checkpoint['model'])

    loss = 0.0
    best_multi_acur = -1.0
    best_binary_acur = -1.0
    args.batch_size = 500
    multiclass_correct = 0
    binary_correct = 0
    multiclass_instance = 0
    binary_instance = 0
    weights = []
    ctx_weights = []
    seq_ids = [k for k in test_data.keys()]
    sort_seq_ids = []
    back_out_method = 'nlap'

    model.eval()
    # save for output labels
    stringOuts = []
    print("Running forward step to extract weights ...")
    # get the loss of test set
    with torch.no_grad():
        # for each epoch do the training
        for sort_feature, sort_targets, seq_len, mask, sort_chunk_ids in \
            generateBatchSST(test_data, test_class, seq_ids, args, batch_size=500):
            # send to device
            mask = mask.to(args.device)
            sort_feature = sort_feature.to(args.device)
            sort_targets = sort_targets.to(args.device)
            # Run forward pass.
            output = model(sort_feature, seq_len, mask)
            # produce readable string encoded results
            stringout = stringOut(sort_targets, output)
            # Weights and collect outputs
            weight = None
            weight, ctx_weight = model.backward_tf_attn(sort_feature, seq_len, mask)
            for i in range(weight.shape[0]):
                weights.append(weight[i])
                ctx_weights.append(ctx_weight[i])
                stringOuts.append(stringout[i])
            sort_seq_ids.extend(sort_chunk_ids)
            multiclass, binary, binary_total = calculate_accuracy(output, sort_targets)
            multiclass_correct += multiclass
            multiclass_instance += len(seq_len)
            binary_correct += binary
            binary_instance += binary_total

    multi_accu = multiclass_correct*1.0/multiclass_instance
    binary_accu = binary_correct*1.0/binary_instance
    logger.info('Test Set Performance\tmulti_acc: {:0.3f} \tbinary_acc: {:0.3f} \t'.\
        format(multi_accu, binary_accu))

    # weights: (b, l)
    all_sentence = pickle.load( open( data_folder + "id_sentence.p", "rb" ) )
    test_sentence = [all_sentence[_id] for _id in sort_seq_ids]
    assert(len(test_sentence) == len(weights))

    id_tf_attns = dict()
    id_ctx_attns = dict()
    for i in range(len(test_sentence)):
        tf_attn = weights[i]
        ctx_attn = ctx_weights[i]
        # adjust the weights length based on sentence length
        s = test_sentence[i]
        id_tf_attns[sort_seq_ids[i]] = tf_attn[:,:,:len(s),:len(s)].tolist()
        id_ctx_attns[sort_seq_ids[i]] = ctx_attn[:len(s)].tolist()

    pickle.dump( id_tf_attns, open(args,out_dir + "/id_tf_attns_sst.p", "wb") )
    pickle.dump( id_ctx_attns, open(out_dir + "/id_ctx_attns_sst.p", "wb") )

def load_token_dict_sst(args):
    data_folder = args.data_dir
    sentence = pickle.load( open( data_folder + "id_sentence.p", "rb" ) )
    # Load phrase mapping
    phrase = dict()
    phrase_file = data_folder + "root/dictionary.txt"
    f = open(phrase_file, "r")
    for x in f:
        seq = " ".join([t.lower() for t in x.split("|")[0].strip().split(" ")])
        phrase_id = int(x.split("|")[1])
        phrase[seq] = phrase_id

    # Load phrase rate mapping
    phrase_rate = dict()
    phrase_rate_file = data_folder + "root/sentiment_labels.txt"
    f = open(phrase_rate_file, "r")
    head = True
    for x in f:
        if head:
            head = False
            continue
        p_id = int(x.split("|")[0])
        rr = float(x.split("|")[1])
        phrase_rate[p_id] = rr

    sentece_token_rate = dict()
    for seq in sentence.keys():
        token_rate_cs = []
        for token in sentence[seq]:
            token_id = phrase[token]
            token_rate = phrase_rate[token_id]
            token_rate_c = -1
            if token_rate >= 0.0 and token_rate <= 0.2:
                token_rate_c = 1
            elif token_rate > 0.2 and token_rate <= 0.4:
                token_rate_c = 2
            elif token_rate > 0.4 and token_rate <= 0.6:
                token_rate_c = 3
            elif token_rate > 0.6 and token_rate <= 0.8:
                token_rate_c = 4
            elif token_rate > 0.8 and token_rate <= 1.0:
                token_rate_c = 5
            else:
                assert(False)
            token_rate_cs.append(token_rate_c)
        sentece_token_rate[seq] = token_rate_cs[:]

    pickle.dump( sentece_token_rate, open(args.out_dir + "/seq_token_rate.p", "wb") )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../../../Stanford-Sentiment-Treebank/",
                        help='path to data base directory')
    parser.add_argument('--model_path', type=str, default="../bd01a5fa/best-model-SST-m.pth",
                        help='path to the saved model (end with .pth)')
    parser.add_argument('--out_dir', type=str, default="../save_lap/",
                        help='the directory to save all the results')
    args = parser.parse_args()

    # These helper script should be combined with others.
    # These are only for SST analysis
    extract_attn_weight(args)
    load_token_dict_sst(args)