#bag of words

import nltk
import pandas as pd
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np

#Opening train file from cvs and assign header
train=pd.read_csv("dev_set.txt",delimiter="\t",names = ["class", "idiom", "text"])
df=pd.DataFrame(train)
train=np.array(df['text'])
stopWords = set(stopwords.words('english'))
wordsFiltered = []

punc=["< b >","< /b > "]
cont=["\'s","\'re","n\'t"]
for i in train:
    temp = ""
    words = word_tokenize(str(i).lower())
    for w in words:
        if w not in stopWords and w not in cont:
            temp= temp + " " + w
    for w in punc:
        temp=temp.replace(w, "")
    wordsFiltered.append(str(temp))

df['text']= wordsFiltered
#print(df['text'])

#CountVectorizer
count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(df['text'])
print(x_train_counts)

#TfidfVectorizer
tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
x_train_tf = tf_transformer.transform(x_train_counts)
print(x_train_tf)


