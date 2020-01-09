import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd

import nltk
nltk.download('punkt')

#subset: choose between train, test or all
#category: for example comp.os.ms-windows.misc
#dimensions: number of relevant words to use
#plot_tf_idf: if true plot the tf-idf summed value for each word
def get_20newsgroup_tf_idf(subset, category, dimensions, plot_tf_idf = False):

    newsgroups_set = fetch_20newsgroups(subset='all', categories=[category])
    newsgroups_set.data = list(set(newsgroups_set.data))
    stop_words = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words = stop_words)
    vectors = vectorizer.fit_transform(newsgroups_set.data)
    ordered_tfidf = np.sort(np.sum(vectors.toarray(), axis = 0))[::-1]
    tf_idf_values = np.sum(vectors.toarray(), axis = 0)
    tf_idf_values_with_index = []
    for i in range(len(tf_idf_values)):
        tf_idf_values_with_index.append((i, tf_idf_values[i]))
    most_importan_words_indexvalues = sorted(tf_idf_values_with_index, key=lambda x: x[1])[::-1][:dimensions]
    most_importan_words_index = list(map(lambda a : a[0], most_importan_words_indexvalues))
    most_importan_words = np.array(vectorizer.get_feature_names())[most_importan_words_index]
    words_to_remove = list(set(vectorizer.get_feature_names()) - set(most_importan_words))
    vectorizer = TfidfVectorizer(stop_words = (list(stop_words) + words_to_remove))
    vectors = vectorizer.fit_transform(newsgroups_set.data)

    matrix_tf_idf = vectors.toarray()
    #print(matrix_tf_idf.shape)
    matrix_tf_idf = matrix_tf_idf/(matrix_tf_idf.sum(axis= 1).reshape(matrix_tf_idf.shape[0], 1))
    #print(matrix_tf_idf.sum(axis = 1))
    if plot_tf_idf:
        X = np.arange(0, len(ordered_tfidf), 1)
        plt.plot(X, ordered_tfidf)
        plt.show()
    return matrix_tf_idf


print(get_20newsgroup_tf_idf("all", "comp.os.ms-windows.misc", 7511, plot_tf_idf = True))
#get_20newsgroup("all", "comp.sys.mac.hardware ")
