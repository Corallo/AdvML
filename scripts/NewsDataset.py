import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.datasets import fetch_20newsgroups

import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

#subset: choose between train, test or all
#categories: list of wanted categories 
#dimensions: number of relevant words to use
#plot_tf_idf: if true plot the tf-idf summed value for each word
def get_20newsgroup_tf_idf(subset, categories, dimensions, plot_tf_idf = False):
    newsgroups_set = fetch_20newsgroups(subset='all', categories=categories)
    newsgroups_set.data = list(set(newsgroups_set.data))
    stop_words = stopwords.words('english')
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
    vectorizer = TfidfVectorizer(stop_words = (stop_words + words_to_remove))
    vectors = vectorizer.fit_transform(newsgroups_set.data)

    matrix_tf_idf = vectors.toarray()
    #print(matrix_tf_idf.shape)
    matrix_tf_idf = matrix_tf_idf/(matrix_tf_idf.sum(axis= 1).reshape(matrix_tf_idf.shape[0], 1))
    #print(matrix_tf_idf.sum(axis = 1))
    if plot_tf_idf:
        X = np.arange(0, len(ordered_tfidf), 1)
        plt.plot(X, ordered_tfidf)
        plt.show()
        
    return matrix_tf_idf, targets, newsgroups_set.target_names

newsgroup_tf_idf, targets, targets_ids = get_20newsgroup_tf_idf("all", ["comp.os.ms-windows.misc", "comp.sys.mac.hardware"], 7511)
