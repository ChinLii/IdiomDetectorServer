#postags: noun,verb,adjective,adverb,determiner

import nltk
import pandas as pd
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.metrics import accuracy_score
import os


def trainingIdiomDetector(filename):
    #Opening datasets file(extracted with POSTAGS) and assign header
    train=pd.read_csv(filename,delimiter="\t",names = ["class", "idiom", "text"])
    test=pd.read_csv(os.path.abspath(os.path.dirname(__file__)) +"/no_class_text_vnaad2.txt",delimiter="\t",names = ["class", "idiom", "text"])
    actual_set=pd.read_csv(os.path.abspath(os.path.dirname(__file__)) +"/test_set.txt",delimiter="\t",names = ["class", "idiom", "text"])


    #CountVectorizer with ngram
    count_vector = CountVectorizer(analyzer="word",           #prerequisite from preprocessor function
                               ngram_range=(1,2),         #obtaining the combination of term as unigram, bigram and trigram
                              )
    x_train_counts = count_vector.fit_transform(train['text'])
    #print(x_train_counts)

    #TfidfVectorizer
    tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
    x_train_tf = tf_transformer.transform(x_train_counts)
    #print(x_train_tf)

    #Transforming a count matrix to a tf representation
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    #Creating a classifier with Linear Support Vector Machine Algorithm
    classifer = svm.LinearSVC(penalty="l2", loss="squared_hinge", dual=False, tol=0.0001,
                            C=5.0, multi_class="ovr", fit_intercept=True, intercept_scaling=1,
                            class_weight=None, verbose=0, random_state=None, max_iter=1000).fit(x_train_tfidf, train['class'])

    #Counting occurences each term in test set
    x_test_counts = count_vector.transform(test['text'])
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)

    #Predicting with a classifier from training set
    predicted = classifer.predict(x_test_tfidf)

    #print(accuracy_score(actual_set['class'], predicted))
    return accuracy_score(actual_set['class'], predicted)
