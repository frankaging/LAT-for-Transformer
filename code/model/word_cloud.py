import os
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import csv

def plot_unsign():
    # sst or unsigned word cloud
    word_score = dict()
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
            word_score[row[0]] = int(float(row[3]) * 100)

    wordcloud = WordCloud(width=1600, height=800).generate_from_frequencies(word_score)
    # Display the generated image:
    # the matplotlib way:
    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig('../../presentation/plots/wordcloud_send.png', facecolor='k', bbox_inches='tight')

plot_unsign()