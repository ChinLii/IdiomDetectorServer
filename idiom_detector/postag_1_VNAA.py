#postags: noun,verb,adjective,adverb

import nltk
import pandas as pd
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from nltk import word_tokenize, pos_tag

#Opening train(development) file and assign header
train=pd.read_csv("dev_set.txt",delimiter="\t",names = ["class", "idiom", "text"])
df=pd.DataFrame(train)
train=np.array(df['text'])
#stopWords = set(stopwords.words('english'))
wordsFiltered = []
punc=["&lsquo;"]
#cont=["\'s","\'re","n\'t"]
#nouns:start with N, verbs:startwith V, adjectives: startwith J, adverbs:startwith R
posttags=["NN", "NNS", "NNP", "NNPS","VB", "VBD", "VBG", "VBN", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"]
for i in train:
    temp = ""
    words = word_tokenize(str(i).lower())
    for w in words:
        c = pos_tag(w)
        if c[0][1] in posttags:
            temp = temp + " " + w
    for w in punc:
        temp = temp.replace(w, "")
    wordsFiltered.append(str(temp))
    temp=temp+"\n"
    print(temp)
    #creating csv file for results of 4 types of word
    with open('result_VNAA.csv', 'a') as file:
        file.write(temp)

#df['text']= wordsFiltered

