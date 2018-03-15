from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re
import pandas as pd


# Function that get the data from the dataset
# I put the data in three diferent arrays because is diferent information that we can get from the dataset
# Tags means the expression, "a cut below" per example
# Classification means the idiom, "literally" per example
# Documents means the actualy sentence
def getDataSet(filename):
    tags = []
    classification = []
    documents = []

    with open(filename, encoding="utf8") as f:
        for line in f:
            tokenize_Sentence = re.split(r'\t+', line)
            classification.append({tokenize_Sentence[2]: tokenize_Sentence[0]})
            tags.append({tokenize_Sentence[1]: tokenize_Sentence[0]})
            documents.append({tokenize_Sentence[0]: tokenize_Sentence[3]})

    return tags, classification, documents


# After loading the document, I think that we will want to look for some expression and search the features in the figuratively or literally and see wich one is better
# But if we don't do the dataset search like that is easy to change
def getDocumentsByTag(tags, classifications, documents, tagName, classificationName):
    documentsByTag = []
    for classification in classifications:
        for keyClas, valueClas in classification.items():
            if (keyClas == classificationName):
                for tag in tags:
                    for keyTag, valueTag in tag.items():
                        if (keyTag == tagName and valueTag == valueClas):
                            for document in documents:
                                for keyDoc, valueDoc in document.items():
                                    if (keyDoc == valueTag):
                                        documentsByTag.append(valueDoc)
    return documentsByTag


# This methods calculate the frequency of the features and show in a table
def createDTM(TestDocuments):
    vect = TfidfVectorizer()
    dtm = vect.fit_transform(TestDocuments)  # create DTM
    # create pandas dataframe of DTM
    return pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())


# This calculates the number os features
def countDocuments(TestDocuments):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(TestDocuments)
    return X_train_counts.shape


# This calculates the frequency of the features
def termFrequecy(TestDocuments, countDocuments):
    tf_transformer = TfidfTransformer(use_idf=False).fit(countDocuments)
    X_train_tf = tf_transformer.transform(countDocuments)
    return X_train_tf.shape


# Here is where I load the information
Tags, Classification, Documents = getDataSet('DataSet/subtask5b_en_allwords_train.txt')

TestDocuments = getDocumentsByTag(Tags, Classification, Documents, "a cut above", "figuratively")

print(createDTM(TestDocuments))

print(countDocuments(TestDocuments))

print(termFrequecy(TestDocuments, countDocuments(TestDocuments)))
