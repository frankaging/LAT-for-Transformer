import os
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image
import random

word_gs = None

proper_noun = ['pakistan',
               'nuvaring',
               'olympia',
               'morocco',
               'sumo',
               'facebook',
               'michigan',
               'a-team',
               'pentathlon',
               'chapman',
               'tonsillitis',
               'pakistan',
               'hopkins',
               'rabbits']

def plot_unsign():
    # sst or unsigned word cloud
    word_score = dict()
    global word_gs
    word_gs_pre = dict()
    with open('../nlap/words_Test_send.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        head = True
        for row in spamreader:
            if head:
                head = False
                continue
            # freq filter here
            if int(row[1]) < 2:
                continue
            if "'" in row[0]: # this is for illustration purposes
                continue
            if row[0] in proper_noun:
                continue
            word_score[row[0]] = int(float(row[3]) * 100)
            word_gs_pre[row[0]] = int(float(row[6]) * 100)

    # construct circle mask
    _mask = np.array(Image.open("../../presentation/plots/word_cloud_mask.png"))

    # let use divide by the max to make sure it is in a unit scale
    # max_b = max([word_score[k] for k in word_score.keys()])
    # for k in word_score.keys():
    #     word_score[k] = (word_score[k]*1.0/max_b)*100

    wordcloud = WordCloud(width=1600, height=800, background_color="white", mask=_mask, max_words=len(word_score), max_font_size=1000, relative_scaling=0).generate_from_frequencies(word_score)
    # Display the generated image:
    # the matplotlib way:
    plt.figure( figsize=(20,16))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig('../../presentation/plots/wordcloud_send.png', bbox_inches='tight')

plot_unsign()