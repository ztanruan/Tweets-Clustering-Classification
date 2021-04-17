
import json
import re
import sys
import operator
from collections import Counter
import nltk
import copy
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
import spacy
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from string import punctuation
import collections
import en_core_web_sm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction import text

# https://github.com/srddy/Sentiment-Analysis-of-Tweets-using-K-Means-Clustering

def jaccard(a, b):
    intersection = list(set(a) & set(b))
    I = len(intersection)
    union = list(set(a) | set(b))
    U = len(union)
    return round(1 - (float(I) / U), 4)

def output(cluster, k, tweet_data):
    final = []
    for i in range(k):
        final.append([j for j, u in enumerate(cluster) if u == i])
        t = [x for x in final[i]]
        print("Cluster [",i+1,"] : ", len([tweet_data[x] for x in t]), " tweets. ")


def kmeans(centroid_index, tweet_data, l, k):
    #count = 0
    for h in range(k):
        #count = count + 1
        centroid_txt = [tweet_data[x] for x in centroid_index]
        cluster = []
        for i in range(l):
            d = [jaccard(tweet_data[i], centroid_txt[j]) for j in range(k)]
            ans = d.index(min(d))
            cluster.append(ans)
        centroid1 = up_date(cluster, tweet_data, l, k)
        sum = 0;
        for i in range(k):
            if(centroid1[i] == centroid_index[i]):
                sum = sum + 1
            if (sum == k):
                break
            centroid_index = copy.deepcopy(centroid1)
    #output(cluster, k, terms_all)
    print("For k  :  ", k)
    sse(cluster, centroid_index, tweet_data, k, l)
    output(cluster, k, tweet_data)


def output(cluster, k, tweet_data):
    final = []
    for i in range(k):
        final.append([j for j, u in enumerate(cluster) if u == i])
        t = [x for x in final[i]]
        print("Cluster [",i+1,"] : ", len([tweet_data[x] for x in t]), " tweets. ")


def up_date(cluster, tweet_data, l, k):
    indices = []
    new_centxt_index = []
    for i in range(k):
        indices.append([j for j, u in enumerate(cluster) if u == i])
        m = indices[i]
        if (len(m) != 0):
            txt = [tweet_data[p] for p in m]
            sim = [[jaccard(txt[i], txt[j]) for j in range(len(m))] for i in range(len(m))]
            f1 = [sum(i) for i in sim]
        new_centxt_index.append(
            m[(f1.index(min([sum(i) for i in sim])))])
    return new_centxt_index


# Error Sum of Squares (SSE) is the sum of the squared differences between each 
# observation and its group's mean. It can be used as a measure of 
# variation within a cluster. If all cases within a cluster are identical 
# the SSE would then be equal to 0.

def sse(cluster, centroid_index, tweet_data, k, l):
    indices1 = []
    centroid_txt = [tweet_data[x] for x in centroid_index]
    sum = 0
    for i in range(k):
        indices1.append([j for j, u in enumerate(cluster) if u == i])
        t = [tweet_data[x] for x in indices1[i]]
        for j in range(len(indices1[i])):
            sum = sum + math.pow(jaccard(t[j], centroid_txt[i]), 2)
    print('SSE', sum)


tweet_data = []

with open('output.json', 'r+') as f:
    students = json.load(f)
    # Print each property of the object
   
    for student in students:
        data = student['text']
        data = data.lower()
        data = re.sub(r'\[.*?\]', '', data)
        data = re.sub(r'[%s]' % re.escape(string.punctuation), '', data)
        data = re.sub(r'\w*\d\w*', '', data)
        data = " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", data).split())

        emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    u"\U00002500-\U00002BEF"  # chinese char
                                    u"\U00002702-\U000027B0"
                                    u"\U00002702-\U000027B0"
                                    u"\U000024C2-\U0001F251"
                                    u"\U0001f926-\U0001f937"
                                    u"\U00010000-\U0010ffff"
                                    u"\u2640-\u2642"
                                    u"\u2600-\u2B55"
                                    u"\u200d"
                                    u"\u23cf"
                                    u"\u23e9"
                                    u"\u231a"
                                    u"\ufe0f"  # dingbats
                                    u"\u3030"
                                    "]+", flags=re.UNICODE)


        data = emoji_pattern.sub(r'', data)
        text_tokens = word_tokenize(data)
        data = [word for word in text_tokens if not word in stopwords.words()]
        data = (" ").join(data)
        data = data.split()
        tweet_data.append(data)


    centroid_index = []
    for i in range(0, len(tweet_data), 1):
        centroid_index.append(i)

    l = len(tweet_data)
    k = 5
    kmeans(centroid_index, tweet_data, l, k)