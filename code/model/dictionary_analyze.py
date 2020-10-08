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
from operator import itemgetter
import pprint
from numpy import newaxis as na

from string import punctuation
import statistics 
import nltk, re

def find_lemma_map(name, word_set, dictionary, bypass_lemma=False):
    '''
    this function will generate a csv file that maps from the word in SEND and 
    the lemmas exist in our dictionary. if no lemma exists in the dictionary,
    we will return a empty string. If multiple appears, we will have a joined
    string separated by comma.
    '''
    from nltk.stem.wordnet import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    from word_forms.word_forms import get_word_forms

    lemma_map = dict()
    for word in word_set:
        # every word will have an empty matching first
        lemma_map[word] = ''
        # first priority is given to those words appear in both sets
        if word in dictionary:
            lemma_map[word] = word
        else:
            if bypass_lemma:
                continue
            # if it does not appear in dictionary, we transform in different ways
            # and see if we can find a close match
            lemma_word = lemmatizer.lemmatize(word)
            if lemma_word in dictionary:
                lemma_map[word] = lemma_word
            else:
                # we will use this last resort of external libraries. the output
                # will then be mannually examined with the output since we only
                # have a couple thousands of words here.
                external_forms = get_word_forms(lemma_word)
                for form in external_forms.keys():
                    words = external_forms[form]
                    for w in words:
                        if w in dictionary:
                            if lemma_map[word] == '':
                                lemma_map[word] = w
                            else:
                                lemma_map[word] = lemma_map[word] + "," + w
    # write the dict to a csv file for mannual examination
    missing_match = 0
    for word in lemma_map.keys():
        if lemma_map[word] == '':
            missing_match += 1
    print('Missing Count: %s, Total Count: %s' % (missing_match, len(lemma_map.keys())))
    output_file = "../warriner_valence/word_lemma_" + name + ".csv"
    with open(output_file, mode='w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',')
        for w in lemma_map.keys():
            row = [w, lemma_map[w]]
            file_writer.writerow(row)

def sanity_check_lemma(lemma_word, dictionary):
    print("====== Checking the matchings in console ======")
    '''
    Step 1:
    check the lemma first, we want to print out those words does not look 
    like the same, i.e. start with the smae characters.
    '''
    print("====== Step 1 ======")
    for key in lemma_word.keys():
        for match in lemma_word[key].split(","):
            if match.strip() != '':
                if key[:2] != match[:2]:
                    print(key.strip(), str(lemma_word[key].strip()))
                    break
    name = input("Press enter to continue next step ... ")
    print("====== Step 2 ======")
    '''
    Step 2:
    check those with multiple matchs if it is ok.
    '''
    for key in lemma_word.keys():
        if len(lemma_word[key].split(",")) > 1:
            print(key.strip(), str(lemma_word[key].strip()))
    name = input("Press enter to continue next step ... ")
    print("====== Step 3 ======")
    '''
    Step 3:
    for all those unmatched words, consider using string matching to find
    matches.
    '''
    lemma_cand = dict()
    for key in lemma_word.keys():
        ks = lemma_word[key].strip()
        if ks == '':
            cands = []
            # at least, we need to have two letters in common, otherwise we will
            # have a lot of noises.
            for i in range(3, len(key)):
                start_str = key[:i]
                refresh = True
                for word in dictionary:
                    if word.startswith(start_str):
                        if refresh:
                            cands = [word]
                            refresh = False
                        else:
                            cands.append(word)
            # if there is no matching, we simply could not find potential match.
            if len(cands) > 1:
                lemma_cand[key] = cands[:]
                print(key, lemma_cand[key])
                print("\n")
    name = input("Press enter to continue next step ... ")
    '''
    Step 4:
    go ahead and modify your correction file mannually to make your match better
    '''

def SST(args):
    dict_file = "../../../Sentiment-Lexicons/Warriner/warriner_valence.csv"
    nlap_file = "../nlap/words_Test_sst.csv"

    dict_score = dict()
    with open(dict_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        head = 1
        for row in readCSV:
            if head:
                head = 0
                continue
            dict_score[row[0]] = float(row[1])

    full_dict_file = "../../../Sentiment-Lexicons/Warriner/Warriner_et_al_emot_ratings.csv"
    dict_std = dict()
    with open(full_dict_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        head = 1
        for row in readCSV:
            if head:
                head = 0
                continue
            dict_std[row[1]] = float(row[3])

    nlap = dict()
    nlap_std = dict()
    word_count = dict()
    gs = dict()
    with open(nlap_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        head = 1
        for row in readCSV:
            if head:
                head = 0
                continue
            word_count[row[0]] = int(row[1])
            nlap[row[0]] = float(row[3])
            nlap_std[row[0]] = float(row[4])
            gs[row[0]] = float(row[6])

    unknown_pre = 0
    for word in set(nlap.keys()):
        if word not in set(dict_score.keys()):
            unknown_pre += 1
    print("Before lemmatization unknown: %s/%s" % (unknown_pre, len(set(nlap.keys())), ))
    '''
    For the first time you can uncomment the following two lines to do lemmatization
    of the words. In this way, you can get the maximum overlap list between the
    dictionary and your word list.
    '''
    # find_lemma_map('sst', set(nlap.keys()), set(dict_score.keys()))
    

    lemma_word = dict()
    lemma_file = "../warriner_valence/word_lemma_sst_correction.csv"
    with open(lemma_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            lemma_word[row[0]] = row[1]
    
    # sanity_check_lemma(lemma_word, dict_score.keys())

    unknown_pre = 0
    for word in set(nlap.keys()):
        if word not in set(dict_score.keys()):
            if word not in lemma_word.keys():
                # this should not happen
                print(word)
            else:
                if lemma_word[word] == '':
                    unknown_pre += 1
    print("After lemmatization unknown: %s/%s" % (unknown_pre, len(set(nlap.keys()))))

    '''
    With these steps, let us produce the final csv file containing both ratings
    from dictionary and the model.
    '''

    output_file = "../warriner_valence/Test_sst.csv"
    with open(output_file, mode='w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',')
        header = ["word", "count", "score", "std", "type", "warriner_valence", "gs"]
        file_writer.writerow(header)
        for word in word_count.keys():
            # only look at those overlapped words
            if lemma_word[word] != '':
                if len(lemma_word[word].split(",")) > 1:
                    match_score = []
                    match_std = []
                    for match in lemma_word[word].split(","):
                        match_score.append(dict_score[match])
                        match_std.append(dict_std[match])
                    avg_score = sum(match_score) * 1.0 / len(match_score)
                    avg_std = sum(match_std) * 1.0 / len(match_std)
                    row = [word, word_count[word], nlap[word], avg_std, "nlap", avg_score, gs[word]]
                    file_writer.writerow(row)
                else:
                    row = [word, word_count[word], nlap[word], dict_std[lemma_word[word]], "nlap", \
                           dict_score[lemma_word[word]], gs[word]]
                    file_writer.writerow(row) 
    print("Writing to file: %s" % (output_file))

def SEND(args):
    dict_file = "../../../Sentiment-Lexicons/Warriner/warriner_valence.csv"
    nlap_file = "../nlap/words_Test_send.csv"

    dict_score = dict()
    with open(dict_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        head = 1
        for row in readCSV:
            if head:
                head = 0
                continue
            dict_score[row[0]] = float(row[1])

    full_dict_file = "../../../Sentiment-Lexicons/Warriner/Warriner_et_al_emot_ratings.csv"
    dict_std = dict()
    with open(full_dict_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        head = 1
        for row in readCSV:
            if head:
                head = 0
                continue
            dict_std[row[1]] = float(row[3])

    nlap = dict()
    nlap_std = dict()
    word_count = dict()
    gs = dict()
    with open(nlap_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        head = 1
        for row in readCSV:
            if head:
                head = 0
                continue
            word_count[row[0]] = int(row[1])
            nlap[row[0]] = float(row[3])
            nlap_std[row[0]] = float(row[4])
            gs[row[0]] = float(row[6])

    unknown_pre = 0
    for word in set(nlap.keys()):
        if word not in set(dict_score.keys()):
            unknown_pre += 1
    print("Before lemmatization unknown: %s/%s" % (unknown_pre, len(set(nlap.keys())), ))
    '''
    For the first time you can uncomment the following two lines to do lemmatization
    of the words. In this way, you can get the maximum overlap list between the
    dictionary and your word list.
    '''
    # find_lemma_map('send', set(nlap.keys()), set(dict_score.keys()))
    # reload and do sanity check
    lemma_word = dict()
    lemma_file = "../warriner_valence/word_lemma_send_correction.csv"
    with open(lemma_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            lemma_word[row[0]] = row[1]
    # sanity_check_lemma(lemma_word, dict_score.keys())

    unknown_pre = 0
    for word in set(nlap.keys()):
        if word not in set(dict_score.keys()):
            if word not in lemma_word.keys():
                # this should not happen
                print(word)
                name = input("Press enter to continue next step ... ")
            else:
                if lemma_word[word] == '':
                    unknown_pre += 1
    print("After lemmatization unknown: %s/%s" % (unknown_pre, len(set(nlap.keys()))))

    output_file = "../warriner_valence/Test_send.csv"
    with open(output_file, mode='w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',')
        header = ["word", "count", "score", "std", "type", "warriner_valence", "gs"]
        file_writer.writerow(header)
        for word in word_count.keys():
            # only look at those overlapped words
            if lemma_word[word] != '':
                if len(lemma_word[word].split(",")) > 1:
                    match_score = []
                    match_std = []
                    for match in lemma_word[word].split(","):
                        match_score.append(dict_score[match])
                        match_std.append(dict_std[match])
                    avg_score = sum(match_score) * 1.0 / len(match_score)
                    avg_std = sum(match_std) * 1.0 / len(match_std)
                    row = [word, word_count[word], nlap[word], avg_std, "nlap", avg_score, gs[word]]
                    file_writer.writerow(row)
                else:
                    row = [word, word_count[word], nlap[word], dict_std[lemma_word[word]], "nlap", \
                           dict_score[lemma_word[word]], gs[word]]
                    file_writer.writerow(row) 
    print("Writing to file: %s" % (output_file))

def ALL(args):
    '''
    this function will look at if the semantics probed by attentions transfer
    between different datasets.
    '''
    send_file = "../nlap/words_Test_send.csv"
    sst_file = "../nlap/words_Test_sst.csv"

    send_nlap = dict()
    send_word_count = dict()
    with open(send_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        head = 1
        for row in readCSV:
            if head:
                head = 0
                continue
            send_word_count[row[0]] = int(row[1])
            send_nlap[row[0]] = float(row[3])

    sst_nlap = dict()
    sst_word_count = dict()
    with open(sst_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        head = 1
        for row in readCSV:
            if head:
                head = 0
                continue
            sst_word_count[row[0]] = int(row[1])
            sst_nlap[row[0]] = float(row[3])

    '''
    since send dataset is much smaller, we will use lemmatized words in send 
    and try to find matches in sst. this will save us for the mannual examination
    times and costs.
    '''
    unknown_pre = 0
    for word in set(send_nlap.keys()):
        if word not in set(sst_nlap.keys()):
            unknown_pre += 1
    print("Before lemmatization unknown: %s/%s" % (unknown_pre, len(set(send_nlap.keys())), ))
    # find_lemma_map('send2sst_fine', set(send_nlap.keys()), set(sst_nlap.keys()), bypass_lemma=True)
    lemma_word = dict()
    lemma_file = "../warriner_valence/word_lemma_send2sst_correction.csv"
    with open(lemma_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            lemma_word[row[0]] = row[1]
    # sanity_check_lemma(lemma_word, sst_nlap.keys())

    unknown_pre = 0
    for word in set(send_nlap.keys()):
        if word not in set(sst_nlap.keys()):
            if word not in lemma_word.keys():
                # this should not happen
                print(word)
                name = input("Press enter to continue next step ... ")
            else:
                if lemma_word[word] == '':
                    unknown_pre += 1
    print("After lemmatization unknown: %s/%s" % (unknown_pre, len(set(send_nlap.keys()))))

    output_file = "../warriner_valence/Test_send2sst.csv"
    with open(output_file, mode='w') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',')
        header = ["word", "count_sst", "count_send", "sst", "send"]

        file_writer.writerow(header)
        for word in send_word_count.keys():
            # only look at those overlapped words
            if lemma_word[word] != '':
                if len(lemma_word[word].split(",")) > 1:
                    match_score = []
                    total_count = 0
                    for match in lemma_word[word].split(","):
                        match_score.append( (sst_nlap[match]*1.0*sst_word_count[match]) )
                        total_count += sst_word_count[match]
                    weighted_avg_score = sum(match_score) * 1.0 / total_count
                    row = [word, total_count, send_word_count[word], weighted_avg_score, send_nlap[word]]
                    file_writer.writerow(row)
                else:
                    row = [word, sst_word_count[lemma_word[word]], send_word_count[word], sst_nlap[lemma_word[word]], send_nlap[word]]
                    file_writer.writerow(row) 
    print("Writing to file: %s" % (output_file))


def main(args):
    if args.dataset == "SST":
        SST(args)
    elif args.dataset == "SEND":
        SEND(args)
    elif args.dataset == "ALL":
        ALL(args)
    else:
        assert(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="SST",
                        help='the dataset we want to run (default: SST)')
    args = parser.parse_args()
    main(args)
