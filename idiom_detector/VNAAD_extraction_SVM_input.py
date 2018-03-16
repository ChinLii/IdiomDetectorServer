#postags: noun,verb,adjective,adverb,determiner

import nltk
import pandas as pd
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize, pos_tag
from sklearn import svm
import os


def predictIdiom(input):
    #Opening train(development) file(extracted with POSTAGS) and assign header
    train=pd.read_csv(os.path.abspath(os.path.dirname(__file__)) +"/VNAAD.txt",delimiter="\t",names = ["class", "idiom", "text"])

    #Getting user input as a test set
    #input=input("Please,Enter the expression you want to know if it includes idiom or not")

    #nouns:start with N, verbs:startwith V, adjectives: startwith J, adverbs:startwith R, Determiner:DT
    posttags=["NN", "NNS", "NNP", "NNPS","VB", "VBD", "VBG", "VBN", "JJ", "JJR", "JJS", "RB", "RBR", "RBS", "DT"]
    test=[]

    #using tokenization and pos tagging
    temp = ""
    words = word_tokenize(str(input).lower())
    for w in words:
        c = pos_tag(w)
        if c[0][1] in posttags:
            temp = temp + " " + w

    #creating the actual test data after preprocessing
    test.append(str(temp))


    #CountVectorizer with ngram
    count_vector = CountVectorizer(analyzer="word",           #prerequisite from preprocessor function
                                   ngram_range=(1,2),         #obtaining the combination of term as unigram, bigram and trigram
                                  )
    x_train_counts = count_vector.fit_transform(train['text'])


    #TfidfVectorizer
    tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
    x_train_tf = tf_transformer.transform(x_train_counts)


    #Transforming a count matrix to a tf representation
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    #Creating a classifier with Linear Support Vector Machine Algorithm
    classifer = svm.LinearSVC(penalty="l2", loss="squared_hinge", dual=False, tol=0.0001,
                            C=5.0, multi_class="ovr", fit_intercept=True, intercept_scaling=1,
                            class_weight=None, verbose=0, random_state=None, max_iter=1000).fit(x_train_tfidf, train['class'])

    #Counting occurences each term in test set
    x_test_counts = count_vector.transform((test))
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)

    #Predicting with a classifier from training set
    predicted = classifer.predict(x_test_tfidf)

    #printing the result from given input
    return predicted

