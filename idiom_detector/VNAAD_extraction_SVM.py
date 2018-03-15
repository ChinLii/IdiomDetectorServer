# postags: noun,verb,adjective,adverb,determiner

import nltk
import pandas as pd
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import numpy as np
from nltk.sentiment.util import mark_negation
from nltk import word_tokenize, pos_tag
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os


# Opening train(development) file(extracted with POSTAGS) and assign header
def Detectidioms(traindata):
    train = pd.read_csv(traindata, delimiter="\t", names=["class", "idiom", "text"])
    test = pd.read_csv(os.path.abspath(os.path.dirname(__file__)) + "/test_set_noclass.txt", delimiter="\t", names=["class", "idiom", "text"])
    actual_set = pd.read_csv(os.path.abspath(os.path.dirname(__file__)) + "/test_set.txt", delimiter="\t", names=["class", "idiom", "text"])

    # CountVectorizer
    count_vector = CountVectorizer(analyzer="word",  # prerequisite from preprocessor function
                                   tokenizer=lambda text: mark_negation(word_tokenize(text)),
                                   # to override  the tokenization with marking negation
                                   ngram_range=(1, 3),
                                   # obtaining the combination of term as unigram, bigram and trigram
                                   )
    x_train_counts = count_vector.fit_transform(train['text'])
    tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
    x_train_tf = tf_transformer.transform(x_train_counts)
    # print(x_train_tf)

    # Transforming a count matrix to a tf representation
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    # Creating a classifier with Naive Bayes Multinomial
    classifer = svm.LinearSVC().fit(x_train_tfidf, train['class'])

    # Counting occurences each term in test set
    x_test_counts = count_vector.transform(test['text'])
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)

    # Predicting with a classifier from training set
    predicted = classifer.predict(x_test_tfidf)
    accuracy = accuracy_score(actual_set['class'], predicted)
    return accuracy


#if __name__ == '__main__':
   # Detectidioms("./uploadFile/dataset.txt")

# print(x_train_counts)

# TfidfVectorizer


"""

#Creating a csv file
with open('VNAAD_result.csv', 'w') as csvfile:
    csvfile.write('idiom,class\n')
    for i,j in zip(test['idiom'], predicted):
        csvfile.write('{},{}\n'.format(i,j))

"""
