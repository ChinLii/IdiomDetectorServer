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

#Opening train(development) file(extracted with POSTAGS) and assign header
train=pd.read_csv("VNAA.txt",delimiter="\t",names = ["class", "idiom", "text"])
df=pd.DataFrame(train)
train=np.array(df['text'])

print(df['text'])

#CountVectorizer
count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(df['text'])
print(x_train_counts)

#TfidfVectorizer
tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
x_train_tf = tf_transformer.transform(x_train_counts)
print(x_train_tf)


