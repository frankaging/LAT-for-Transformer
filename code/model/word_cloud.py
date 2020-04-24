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

    wordcloud = WordCloud(background_color="white").generate_from_frequencies(word_score)
    # Display the generated image:
    # the matplotlib way:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

plot_unsign()