# -*- coding: utf-8 -*-

#!pip install MulticoreTSNE
#!pip install git+https://github.com/crew102/validclust.git

import IPython
IPython.display.clear_output()

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random
from datetime import datetime

from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d
from sklearn.decomposition import PCA, LatentDirichletAllocation
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from validclust import dunn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

"""# Data loading and preprocessing"""

# Please put a number 10 < N < 50000. Entering 50000 means using all tweets.
# I do not recommend using all tweets on a low-end computers, since it will take
# a seriously long time.

N = 5000

with open('output.json', 'r') as f:
    tweets_json = json.loads(f.read())
    tweets = [t['text'].lower() for t in tweets_json]

    idx = list(range(len(tweets)))
    random.shuffle(idx)
    idx = idx[:N]
    tweets = np.array(tweets)[idx]

"""# Encoding"""

# TF features for all algorithms except NMF
tf_vectorizer = CountVectorizer(max_df=0.95, 
                                min_df=2, 
                                max_features=100, 
                                stop_words='english')
tf = tf_vectorizer.fit_transform(tweets)

# TF-IDF features for NMF
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=100,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(tweets)

"""# Clustering
We will find cluster labels for each algorithm

## 1.a. K-Means
"""

from sklearn.cluster import KMeans
kmeans = KMeans(3).fit(tf.toarray())
y_kmeans = kmeans.labels_

np.save('y_kmeans.npy', y_kmeans)

tf_train, tf_test, _, _ = train_test_split(tf, tf, test_size=0.2)

kmeans_split = KMeans(3).fit(tf_train.toarray())
y_kmeans_train = kmeans_split.predict(tf_train)
y_kmeans_test = kmeans_split.predict(tf_test)

np.save('y_kmeans_train.npy', y_kmeans_train)
np.save('y_kmeans_test.npy', y_kmeans_test)

"""## 2. Mean Shift (warning: slow!)"""

from sklearn.cluster import MeanShift
y_mean_shift = MeanShift(n_jobs=-1).fit_predict(tf.toarray())
np.save('y_mean_shift.npy', y_mean_shift)

"""## 3. Agglomerative"""

from sklearn.cluster import AgglomerativeClustering
y_agglomerative = AgglomerativeClustering().fit_predict(tf.toarray())
np.save('y_agglomerative.npy', y_agglomerative)

"""## 4. DBSCAN"""

from sklearn.cluster import DBSCAN
dbscan = DBSCAN().fit(tf.toarray())
y_dbscan = dbscan.labels_ + 1
np.save('y_dbscan.npy', y_dbscan)

"""## 5. EM Gaussian mixture model"""

from sklearn.mixture import GaussianMixture
y_gmm = GaussianMixture(3).fit_predict(tf.toarray())
np.save('y_gmm.npy', y_gmm)

"""## 6. LDA"""

lda = LatentDirichletAllocation(n_components=3, n_jobs=-1)
lda.fit(tf)
y_lda = lda.transform(tf).argmax(1)

np.save('y_lda.npy', y_lda)

"""# 7. NMF

Different from the other algorithms, NMF requires term frequency–inverse (TF-IDF) features. Thus, a different processing pipeline is appliad on this algorithm.
"""

from sklearn.decomposition import NMF
nmf = NMF(n_components=3).fit(tfidf)
y_nmf = nmf.transform(tfidf).argmax(1)

np.save('y_nmf.npy', y_nmf)

"""# Evaluation"""

# As for dunn index, we need it to make outselves.
# Not from scratch, as it depends on validclust package
def dunn_index(points, labels):
    dists = euclidean_distances(points, points)
    return dunn(dists, labels)

data_list = [tf.toarray(),
             tf_test.toarray(), 
             tf.toarray(),
             tf.toarray(),
             tf.toarray(),
             tf.toarray(),
             lda.transform(tf),
             nmf.transform(tfidf)]

labels = [y_kmeans,
          y_kmeans_test,
          y_mean_shift,
          y_agglomerative,
          y_dbscan,
          y_gmm,
          y_lda,
          y_nmf]
          
model_names = ['K-Means', 'K-Means (split)', 'Mean Shift', 'Agglomerative', 'DBSCAN', 'EM GMM', 'LDA', 'NMF']
scoring_names = ['davies bouldin index', 'dunn index', 'silhouette coefficient']
scoring_funcs = [davies_bouldin_score, dunn_index, silhouette_score]

"""## Score assessment

Notes:
- Davies bouldin index: minimum value is zero, **smaller is better**
- Dunn index: minimum value is zero, **higher is better**
- Silhouette coefficient: minimum value is -1 and maximum value is 1, **higher is better**
"""

# Compute scores and populate the result table
score_table = pd.DataFrame(index=model_names, columns=scoring_names)
for model_name, data, label in zip(model_names, data_list, labels):
    for scoring_name, scoring_func in zip(scoring_names, scoring_funcs):
        if not((data is None) or (label is None)):
            score = scoring_func(data, label)
            score_table[scoring_name][model_name] = score
score_table

"""## Score plots"""

score_table.plot.bar()

# Save the table
score_table.to_csv('assessment.csv')

"""## Visualization
Tweets are high-dimension data. In order to visualize the tweets corresponding to the clustering result, we need to reduce the dimensionality. In this case, I use PCA for the sake of efficiency.
"""

def show_plot(title, input_features, labels, use_3d=False):
    dim = 3 if use_3d else 2
    x = PCA(dim).fit_transform(input_features)
    fig = plt.figure()
    ax = plt.axes(projection='3d') if use_3d else plt.axes()
    ax.scatter(*x.T, c=labels, s=50, alpha=0.5)
    plt.title(title)
    plt.show()

for model_name, data, label in zip(model_names, data_list, labels):
    if not(data is None):
        show_plot(model_name, data, label, use_3d=True)

"""## Extra visualization for K-means (split)"""

# Reduce the dimension of both training and testing data for plotting
dim = 3
x_train = PCA(dim).fit_transform(tf_train.toarray())
x_test = PCA(dim).fit_transform(tf_test.toarray())

# Scatter plotting
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(*x_train.T, c=y_kmeans_train, s=20, alpha=1, marker='o', label='Training samples')
ax.scatter(*x_test.T, c=y_kmeans_test, s=50, alpha=1, marker='x', label='Testing samples')

# The appearance setting
ax.legend(loc=4)
leg = ax.get_legend()
leg.legendHandles[0].set_color('black')
leg.legendHandles[1].set_color('black')
plt.title('K-Means (split)')
plt.show()

