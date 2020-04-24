import numpy as np

latex_special_token = ["!@#$%^&*()"]

def generate(text_list_t, attention_list_t, labels_t, latex_file, color='red', rescale_value = False):
    with open(latex_file,'w') as f:
        f.write(r'''
        \begin{CJK*}{UTF8}{gbsn}'''+'\n')
        sentence_num = len(text_list_t)
        for i in range(sentence_num):
            text_list = clean_word(text_list_t[i])
            attention_list = attention_list_t[i]
            labels = labels_t[i]
            word_num = len(text_list)
            string = r'''True Label: ''' + labels[0] + r''', Predicted Label: ''' + labels[1] + r'''\\'''+"\n"
            string += r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
            for idx in range(word_num):
                string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + text_list[idx]+"} "
            string += "\n}}}"
            string += r'''\\ \\'''
            f.write(string+'\n')
        f.write(r'''\end{CJK*}''')

def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)/(the_max-the_min)*100
	return rescale.tolist()


def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list


def SST():
    import pickle
    import random
    id_weights = pickle.load( open( "../nlap/id_weights_test_sst.p", "rb" ) )
    id_labels = pickle.load( open( "../nlap/id_labels_test_sst.p", "rb" ) )
    data_folder = "../../../Stanford-Sentiment-Treebank/"
    all_sentence = pickle.load( open( data_folder + "id_sentence.p", "rb" ) )

    very_positive = []
    very_negative = []
    for _id in id_labels.keys():
        if id_labels[_id][0] == "$++$":
            very_positive.append(_id)
        elif id_labels[_id][0] == "$--$":
            very_negative.append(_id)


    very_positive = random.sample(very_positive, 15)
    very_negative = random.sample(very_negative, 15)

    weights = [id_weights[i] for i in very_positive]
    labels = [id_labels[i] for i in very_positive]
    sentences = [all_sentence[i] for i in very_positive]

    for i in very_negative:
        weights.append(id_weights[i])
        labels.append(id_labels[i])
        sentences.append(all_sentence[i])

    weights_max = []
    for w in weights:
        weights_max.append([ ((i*1.0 - min(w))/(max(w) - min(w)))*100.0 for i in w ])
    
    color = 'red'
    generate(sentences, weights_max, labels, "../nlap/unsigned_sst.tex", color)

def SEND():
    import pickle
    import random
    weights_plot = pickle.load( open( "../nlap/seq_weights_test_send.p", "rb" ) )
    labels_plot = pickle.load( open( "../nlap/seq_labels_test_send.p", "rb" ) )
    sentence_plot = pickle.load( open( "../nlap/seq_sentences_test_send.p", "rb" ) )

    # only use the first one (picked by ccc)
    weights = weights_plot['165_4']
    labels = labels_plot['165_4']
    s = sentence_plot['165_4']
    
    weights_max = []
    for w in weights:
        weights_max.append([ ((i*1.0 - min(w))/(max(w) - min(w)))*100.0 for i in w ])

    labels_process = []
    for l in labels:
        if l[0] > 0:
            left = str(l[0])[:5]
        elif l[0] < 0:
            left = str(l[0])[:6]

        if l[1] > 0:
            right = str(l[1])[:5]
        elif l[1] < 0:
            right = str(l[1])[:6]   
        labels_process.append(['$' + left + '$', '$' + right + '$'])

    color = 'red'
    generate(s, weights_max, labels_process, "../nlap/unsigned_send.tex", color)

if __name__ == '__main__':
    SST()
    # SEND()