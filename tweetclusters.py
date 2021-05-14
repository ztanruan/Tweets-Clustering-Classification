#!pip install vaderSentiment


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import time
import time
from numpy import mean
import nltk
import regex as re
import warnings
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.stem.wordnet import WordNetLemmatizer 
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from pylab import *
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from time import time
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
stop_words = stopwords.words('english')
warnings.filterwarnings("ignore")



def data_cleaning(text):
    
    text = re.sub('[^a-zA-Z]', ' ', text) 
    text = re.sub('(^\s+|\s+$)', ' ', text) 
    text = re.sub("@[\w\d]+", ' ', text)           #delete any references to other people
    text = re.sub("http:[\w\:\/\.]+",' ', text)    #replace url's
    text = re.sub('[^[A-Za-z]\s]',' ', text)      #replace non alphabets and non spaces
    text = text.lower()
    return text

    
def data_tokenization(text):
  lem = WordNetLemmatizer()
  tokens = nltk.tokenize.word_tokenize(text)
  tokens = [token if len(token)>1 else token.replace(token,' ') for token in tokens ]
  tokens = [token for token in tokens if not token in stop_words]
  tokens = [lem.lemmatize(token) for token in tokens]
  tokens = ' '.join(tokens)
  return tokens


def data_preparation(df):
  df["clean_text"] = df["text"].map(lambda x: data_cleaning(x))
  df["clean_texts_token"] = df["clean_text"].map(lambda x: data_tokenization(x))


    
def Kmeans(df):
  # Import module
    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=1000)
    # Fit to the data and transform to tf-idf
    vectorizer.fit_transform(df['clean_texts_token'])
    vector = vectorizer.transform(df["clean_texts_token"])
    cc = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names(), index= df.index)
    cc.reset_index(drop=True, inplace=True)
    #Kmeans Algorthim
    X = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names(), index= df.index)
    X.reset_index(drop=True, inplace=True)
    cls = MiniBatchKMeans(n_clusters=5, random_state=40)
    cls.fit(X)
    y_pred = cls.predict(X)
    # reduce the features to 2D
    pca = PCA(n_components=2, random_state=40)
    reduced_features = pca.fit_transform(features)
    
    # reduce the cluster centers to 2D
    reduced_cluster_centers = pca.transform(cls.cluster_centers_)
    plt.scatter(reduced_features[:,0], reduced_features[:,1], c=cls.predict(features))
    #plt.close('all')
    #plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
    print("-----------------KMEANS------------------")
    print(davies_bouldin_score(X, y_pred))
    print(silhouette_score(X, y_pred))
    

def AgglomerativeClustering(df):

    from sklearn.cluster import AgglomerativeClustering
      # Import module
    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=1000)
    # Fit to the data and transform to tf-idf
    vectorizer.fit_transform(df['clean_texts_token'])
    vector = vectorizer.transform(df["clean_texts_token"])
    cc = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names(), index= df.index)
    cc.reset_index(drop=True, inplace=True)
    #Kmeans Algorthim
    X = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names(), index= df.index)
    X.reset_index(drop=True, inplace=True)
    AgglomerativeClustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    y_predA = AgglomerativeClustering.fit_predict(X)
    # reduce the features to 2D
    pca = PCA(n_components=2, random_state=40)
    reduced_features = pca.fit_transform(features)
    print("-----------------AgglomerativeClustering------------------")
    # reduce the cluster centers to 2D
    #reduced_cluster_centers = pca.transform(AgglomerativeClustering.cluster.labels_)

    plt.scatter(reduced_features[:, 0],y_predA, c=AgglomerativeClustering.labels_, cmap='rainbow')
    #plt.close('all')
    #plt.scatter(reduced_features[:,0], reduced_features[:,1], c=AgglomerativeClustering.predict(features))
    #plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
    print(davies_bouldin_score(X, y_predA))
    print(silhouette_score(X, y_predA))

def DBSCAN(df):
    from sklearn import metrics
    from sklearn.datasets import make_circles
    from sklearn.cluster import DBSCAN
    # Compute DBSCAN
    db = DBSCAN().fit(X)
    labels = db.labels_
    no_clusters = len(np.unique(labels) )
    no_noise = np.sum(np.array(labels) == -1, axis=0)
    # Generate scatter plot for training data
    colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
    pca = PCA(n_components=2, random_state=40)
    reduced_features = pca.fit_transform(features)
    plt.scatter(reduced_features[:,0], reduced_features[:,1], c=colors, marker="o", picker=True)
    plt.show()


def meanshift(df):
    from sklearn.cluster import MeanShift
    meanshift = MeanShift()
    meanshift.fit(X)
    labels = meanshift.labels_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    # Predict the cluster for all the samples
    P = meanshift.predict(X)
    pca = PCA(n_components=2, random_state=40)
    reduced_features = pca.fit_transform(features)
    plt.scatter(reduced_features[:,0], reduced_features[:,1], c=meanshift.predict(features))
    plt.show()

def data_pipeline(df):
  df["clean_text"] = df["text"].map(lambda x: data_cleaning(x))
  df["clean_texts_token"] = df["clean_text"].map(lambda x: data_tokenization(x))
  Kmeans(df)
  DBSCAN(df)
  AgglomerativeClustering(df)
  meanshift(df)


def main():
    #put your file and code in same file
    dfs =pd.read_json ('output.json')
    df = dfs.dropna(axis=0)
    df.drop(['user','target','id','date','flag'], axis = 1,inplace = True)
    data_pipeline(df)
    df.drop(['text','clean_text'], axis = 1,inplace = True)
   
if __name__ == "__main__":
      main()




