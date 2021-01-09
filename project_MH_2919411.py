#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module contains the script for 'Title Clusters in Vector Space'."""
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
import numpy as np  # scientific computing
import pandas as pd
import random  # random number
import re
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
from sklearn import cluster
from sklearn import metrics
from tqdm import tqdm  # progress bar
# NLTK
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.cluster import KMeansClusterer
from nltk.corpus import words  # Unix word corpus (/usr/share/dict/words)
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# Word2Vec
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
# GloVe
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

WORDLIST = {word.lower() for word in words.words()}  # nonword_detector()
stopWords = set(stopwords.words('english'))  # stopword_filter()
porter_stemmer = PorterStemmer()  # get_stem()
WORD2STEM = {}
STEMS = []

glove_path = "/home/marx/glove.6B/glove.6B.300d.txt"  # insert path to file
data_path = "collection-master/artwork_data.csv"
plt_f = "plt/"  # folder for plots
df_f = "df/"  # folder for data frames

# initialise Principal Component Analysis
pca_3d = PCA(n_components=3, random_state=42)


# MAIN_DF
def tokenize_title(title_string):
    """
    Tokenize title and remove all non-words
    param1: title string
    output: title string of joined word tokens
    """
    # lower
    title_string = title_string.lower()
    # normalize apostrophes
    title_string = title_string.replace("’s ", "\'s ")
    # remove comma in numbers
    title_string = re.sub(r"(?<=\d),(?=\d)", "", title_string)
    # remove punctuation marks (except for apostrophe)
    not_word_pattern = re.compile('[^a-zäöüßáàâéèêíìî0-9\']+')
    title_string = not_word_pattern.sub(' ', title_string)
    # tokenize
    title_tokens = word_tokenize(title_string)
    tokens = [token for token in title_tokens]
    # no empty titles
    if len(tokens) < 1:
        return False
    return ' '.join(tokens)


def stopword_filter(words):
    """
    Filter stop words
    param1: list of words
    output: filtered list of words / None
    """
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w)
    if len(wordsFiltered) == 0:
        return None
    else:
        return wordsFiltered


# WORDS_DF
def get_stem(word):
    '''
    Extract stem of the given word
    param1: original word
    output: extracted stem
    '''
    if word not in WORD2STEM:
        WORD2STEM[word] = porter_stemmer.stem(word)  # dict
        STEMS.append(WORD2STEM[word])  # list
    return WORD2STEM[word]


def get_lemma(word, tag):
    '''
    Normalize nouns and verbs
    param1: original word
    param2: POS tag
    output: extracted stem
    '''
    pos_dict = {"N": "n", "V": "v", "J": "a", "R": "r"}
    for pos in pos_dict:
        if tag.startswith(pos):
            lemma = WordNetLemmatizer().lemmatize(word, pos_dict[pos])
            return lemma
        else:
            return word


def nonword_detector(word):
    '''
    Identify non-words using words and their stems
    param1: stem of the word
    param2: original word
    output: binary decision whether the word is a real English word
    '''
    if word in WORDLIST:
        return False
    else:
        return True


def stopword_detector(word):
    '''
    Identify non-words using words and their stems
    param1: stem of the word
    param2: original word
    output: binary decision whether the word is a real English word
    '''
    if word in stopWords:  # set
        return True
    return False


# WORD2VEC / GLOVE
def scatterplot3D(model, color="black", view=None, output=None):
    """
    Display 3-D scatterplot
    param1: model (word2vec)
    param2: color (labels) for data points
    param3: tupel of elevation angle (z) and azimuth angle (x,y)
    param4: name (path) of output file
    output: shows/saves plot
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(model[:, 0], model[:, 1], model[:, 2],
               c=color, s=1, alpha=0.15, marker=".")
    # view
    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])  # elev: z, azim: x, y
    # save
    if output is not None:
        plt.savefig(plt_f+output, bbox_inches="tight")


def sent_vectorizer(sent, model):  # try except statement removed
    """
    Vectorize sentence (average of word vectors)
    param1: sentence (list)
    param2: vector model
    output: average sentance vector (array)
    """
    sent_vec = []
    numw = 0
    for w in sent:
        if numw == 0:
            sent_vec = model[w]  # numpy.ndarray
        else:
            sent_vec = np.add(sent_vec, model[w])
        numw += 1
    return np.asarray(sent_vec) / numw  # average


def nltk_inertia(feature_matrix, centroid):
    """
    Calculate inertia (within cluster sum of squares /
    sum of suqared distances of samples to their closest cluster center)
    param1: feature matrix (array of list of lists [vectors])
    param2: centroids (array of list of arrays [vectors])
    output: inertia
    source: https://stackoverflow.com/questions/59549953/
            get-inertia-for-nltk-k-means-clustering-using-cosine-similarity
    """
    sum_ = []
    for i in range(feature_matrix.shape[0]):
        # implement inertia
        sum_.append(np.sum((feature_matrix[i] - centroid[i])**2))
    return sum(sum_)


def SilhouetteScores(vectors, mode, cluster_range, view, output):
    """
    Compute silhouette and elbow method (nltk/sklearn) for range of clusters
    param1: array of vecotors (3 dimensions)
    param2: mode ('nltk'/'sklearn'/'agglomerative')
    param3: list/range of number of clusters
    param4: output path of figure (needs {} to format number of clusters)
    output: tupel of two dicts with elbow and silhouette scores (nltk/sklearn),
            dict with silhouette scores (agglomerative)
    """
    if mode.lower() == "nltk":
        rng = random.Random()
        rng.seed(123)
        wcss = {}  # for elbow method
        s_scores = {}  # for silhouette scores
        for NUM_CLUSTERS in tqdm(cluster_range):
            kclusterer = KMeansClusterer(NUM_CLUSTERS,
                                         distance=nltk.cluster.util.cosine_distance,
                                         repeats=25,
                                         rng=rng,
                                         avoid_empty_clusters=True)
            labels = kclusterer.cluster(vectors, assign_clusters=True)
            # elbow method
            # the centroids: kclusterer.means()
            centroid_array = np.vstack([kclusterer.means()[label]
                                        for label in labels])
            wcss[NUM_CLUSTERS] = nltk_inertia(vectors, centroid_array)
            # silhouette scores
            if 1 < NUM_CLUSTERS:
                silhouette_s = metrics.silhouette_samples(vectors, labels,
                                                          metric='cosine')
                ss_max = max(silhouette_s)
                ss_min = min(silhouette_s)
                ss_mean = float(sum(silhouette_s)/len(silhouette_s))
                s_scores[NUM_CLUSTERS] = (ss_max, ss_min, ss_mean)
                # plotting
                scatterplot3D(vectors,
                              color=labels,
                              view=view,
                              output=output.format(NUM_CLUSTERS))
        return (wcss, s_scores)

    elif mode.lower() == "sklearn":
        wcss = {}  # for elbow method
        s_scores = {}  # for silhouette scores
        for NUM_CLUSTERS in tqdm(cluster_range):
            kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS, n_init=25,
                                    random_state=42)
            kmeans.fit(vectors)  # compute k-means clustering
            labels = kmeans.labels_
            # elbow method
            wcss[NUM_CLUSTERS] = kmeans.inertia_
            # silhouette scores
            if 1 < NUM_CLUSTERS:
                silhouette_s = metrics.silhouette_samples(vectors, labels,
                                                          metric='euclidean')
                ss_max = max(silhouette_s)
                ss_min = min(silhouette_s)
                ss_mean = float(sum(silhouette_s)/len(silhouette_s))
                s_scores[NUM_CLUSTERS] = (ss_max, ss_min, ss_mean)
                # plotting
                scatterplot3D(vectors,
                              color=labels,
                              view=view,
                              output=output.format(NUM_CLUSTERS))
        return (wcss, s_scores)

    elif mode.lower() == "agglomerative":
        s_scores = {}
        for NUM_CLUSTERS in tqdm(cluster_range):
            agglo = cluster.AgglomerativeClustering(n_clusters=NUM_CLUSTERS)
            agglo.fit(vectors)  # compute k-means clustering
            labels = agglo.labels_
            # silhouette scores
            if 1 < NUM_CLUSTERS:
                silhouette_s = metrics.silhouette_samples(vectors, labels,
                                                          metric='euclidean')
                ss_max = max(silhouette_s)
                ss_min = min(silhouette_s)
                ss_mean = float(sum(silhouette_s)/len(silhouette_s))
                s_scores[NUM_CLUSTERS] = (ss_max, ss_min, ss_mean)
                # plotting
                scatterplot3D(vectors,
                              color=labels,
                              view=view,
                              output=output.format(NUM_CLUSTERS))
        return s_scores


def plot_elbow(e_scores, output=None):
    """
    Plot elbow method
    param1: dict of elbow method scores
    param2: path of output file
    output: shows/saves plot
    """
    cluster_range = list(e_scores.keys())
    fig = plt.figure(figsize=(12, 9))
    plt.plot(list(e_scores.keys()), list(e_scores.values()), color="black")
    plt.xticks(np.arange(cluster_range[0], cluster_range[-1]+1, 1.0))
    if output is not None:
        plt.savefig(plt_f+output, bbox_inches="tight")
#    plt.show()


def plot_silhouette(s_scores, output=None):
    """
    param1: dict of silhouette scores
    param2: path of output file
    output: shows/saves plot
    """
    cluster_range = list(s_scores.keys())
    plt.figure(figsize=(12, 9))
    max_vals = [score[0] for score in s_scores.values()]
    min_vals = [score[1] for score in s_scores.values()]
    mean_vals = [score[2] for score in s_scores.values()]
    plt.plot(list(s_scores.keys()), max_vals,
             c="green", marker="|", label='max. value')
    plt.plot(list(s_scores.keys()), mean_vals,
             c="black", marker="|", label='mean score')
    plt.plot(list(s_scores.keys()), min_vals,
             c="red", marker="|", label='min. value')
    plt.xticks(np.arange(cluster_range[0], cluster_range[-1]+1, 1.0))
    plt.legend()
    if output is not None:
        plt.savefig(plt_f+output, bbox_inches="tight")
#    plt.show()


# GLOVE
def title_filter(title, model):
    """
    Detect out-of-vocabulary titles
    param1: title (list of words)
    param2: model (word2vec format)
    param3: threshold (percentage; 0 to 1)
    output: True (in vocabulary) / False (out-of-vocabulary)
    """
    for word in title:
        if word not in model.vocab:
            return False
    return True


# MAIN_DF
# for 'year' in 'converters'; transform non numeric values to zero
main_df = pd.read_csv(data_path,
                      usecols=["id", "artist", "artistId", "title", "year"],
                      dtype={"id": int, "artist": object,
                             "artistId": int, "title": object},
                      converters={"year": (lambda x: 0 if not x.isdigit()
                                           else int(x))})
# set all missing values for title as 'NaN' (and remove them)
regex_no_title = re.compile("\[title not known\]|"
                            "\[?blank\]?|"
                            "untitled|"
                            "no title", re.IGNORECASE)
main_df = main_df.replace(to_replace=regex_no_title, value=np.nan, regex=True)
main_df = main_df.dropna(subset=["title"])  # remove rows without title
# preprocessing of titles
main_df["title"] = main_df.title.apply(lambda x: str(tokenize_title(x)))
tok_titles = main_df.title.str.split().tolist()
main_df["title_split"] = tok_titles
main_df["title_sw"] = main_df["title_split"].apply(lambda x:
                                                   stopword_filter(x))
main_df["title_split_len"] = main_df["title_split"].apply(lambda x: len(x))
main_df["title_sw_len"] = main_df["title_sw"].apply(lambda x: int(len(x))
                                                    if x is not None else 0)
main_df.to_csv(df_f+"main_df.csv", index=False)  # save data frame
print("main_df.csv created")

# WORDS_DF
# collect data and create data frame
rows = list()
for row in main_df[["id", "title_split",
                    "artist", "year", "title_sw"]].iterrows():
    r = row[1]
    for word in r.title_split:
        if type(r.title_sw) == list:  # not NaN changed FROM STR TO LIST
            title_sw = True
        else:  # is NaN
            title_sw = False
        rows.append((r.id, word, r.artist, r.year, title_sw))
words_df = pd.DataFrame(rows, columns=["id", "word", "artist", "year",
                                       "valid_title_sw"])
# (POS) tag column
tagged_titles = main_df["title_split"].apply(lambda x: pos_tag(x))
tagged_words = list(itertools.chain.from_iterable(tagged_titles))
words_df["tag"] = [tag[1] for tag in tagged_words]
# stem column
words_df["stem"] = words_df["word"].apply(lambda x: get_stem(x))
# lemma column
words_df["lemma"] = words_df.apply(lambda x: get_lemma(x.word, x.tag), axis=1)
# nonword column
words_df["nonword"] = words_df["lemma"].apply(lambda x: nonword_detector(x))
# stopword column
words_df["stopword"] = words_df["word"].apply(lambda x: stopword_detector(x))
# save data frames
words_df.to_csv(df_f+"words_df_large.csv", index=False)
words_valid = words_df[words_df["valid_title_sw"] == True].reset_index()
del words_valid["valid_title_sw"]
words_valid.to_csv(df_f+"words_df.csv", index=False)
print("words_df.csv created")

# plot figure
mpl.style.use('ggplot')  # set style
mpl.rcParams['figure.figsize'] = (12,6)
mpl.rcParams['font.size'] = 12
mpl.rcParams['grid.color'] = 'w'
mpl.rcParams['patch.facecolor'] = 'b'
no_sw_series = words_df.loc[words_df["stopword"] == False, "word"]
this_slice = 25
plt.tick_params(labelsize=22)
no_sw_series.value_counts()[:this_slice].plot.bar(color="black")
plt.savefig(plt_f+"words-sw_frequency_20.png", bbox_inches="tight")
mpl.rcParams.update(mpl.rcParamsDefault)  # reset default style

# READ SAVED FRAME
#main_df = pd.read_csv(df_f+"main_df.csv", na_filter = False)
#words_df = pd.read_csv(df_f+"words_df.csv", na_filter = False)
# del rows with  empty title_sw
#main_df = main_df[main_df['title_sw'] != ""].reset_index()
main_df = main_df.dropna().reset_index()

# WORD2Vec
tok_titles = main_df["title_sw"].tolist()
model = Word2Vec(tok_titles, min_count=1, seed=42, iter=100)
M_sw = model.wv  # 100 dimensions
del model
M_sw_3d = pca_3d.fit_transform(M_sw.vectors)  # 3 dimensions

scatterplot3D(M_sw_3d, view=(60, 40),
              output="word2vec_vocabulary.png")

X_sw = []
for title in tok_titles:
    X_sw.append(sent_vectorizer(title, M_sw))
X_sw_3d = pca_3d.fit_transform(X_sw)

main_df[['feature1', 'feature2', 'feature3']] = pd.DataFrame(X_sw_3d)
vectors_3d = np.array([feature for feature in
                       main_df[['feature1', 'feature2', 'feature3']].values])

scatterplot3D(vectors_3d, view=(60, 40),
              output="word2vec_titles.png")

# K-MEANS: NLTK (cos)
output_path = "word2vec_nltk-cos_{}-clusters.png"  # format
scores_nltk = SilhouetteScores(vectors_3d,
                               mode="nltk",
                               cluster_range=range(1, 7),
                               view=(60, 40),
                               output=output_path)
plot_elbow(scores_nltk[0],
           output="word2vec_nltk-cos_elbow-method.png")
plot_silhouette(scores_nltk[1],
                output="word2vec_nltk-cos_silhouette-scores.png")
rng = random.Random()
rng.seed(123)
NUM_CLUSTERS = 2
kclusterer = KMeansClusterer(NUM_CLUSTERS,
                             distance=nltk.cluster.util.cosine_distance,
                             repeats=25, rng=rng)  # clustering
main_df['cluster_kmeans'] = kclusterer.cluster(vectors_3d,
                                               assign_clusters=True)
main_df['centroid_kmeans'] = main_df['cluster_kmeans'].apply(lambda x:
                                                             kclusterer.means()[x])
print("k-means clustering (NLTK) completed (Word2Vec)")

# K-MEANS: SCI-KIT LEARN (euc)
output_path = "word2vec_sklearn-euc_{}-clusters.png"
scores_sklearn = SilhouetteScores(vectors_3d,
                                  mode="sklearn",
                                  cluster_range=range(1, 7),
                                  view=(60, 40),
                                  output=output_path)
plot_elbow(scores_sklearn[0],
           output="word2vec_sklearn-euc_elbow-method.png")
plot_silhouette(scores_sklearn[1],
                output="word2vec_sklearn-euc_silhouette-scores.png")
print("k-means clustering (sklearn) completed (Word2Vec)")

# DBSCAN
main_df["is_duplicate"] = main_df["title_sw"].apply(lambda x:
                                                    str(x)).duplicated()
duplicates_df = main_df[main_df["is_duplicate"] == False]
vectors_3d_clean = np.array([feature for feature in
                             duplicates_df[["feature1",
                                            "feature2",
                                            "feature3"]].values])
dbs = cluster.DBSCAN(eps=0.66, min_samples=202).fit(vectors_3d_clean)
labels = dbs.labels_
main_df.loc[main_df['is_duplicate'] == False, 'cluster_dbscan'] = dbs.labels_
scatterplot3D(vectors_3d_clean,
              color=labels,
              view=(60, 40),
              output="word2vec_dbscan_clusters.png")
print("DBSCAN clustering completed (Word2Vec)")

# AGGLOMERATIVE CLUSTERING
main_df_slice = main_df.sample(n=20000, random_state=42)
vectors_3d_slice = np.array([feature for feature in
                             main_df_slice[["feature1",
                                            "feature2",
                                            "feature3"]].values])
output_path = "word2vec_agglomerative_{}-clusters.png"
scores_agglomerative = SilhouetteScores(vectors_3d_slice,
                                        mode="agglomerative",
                                        cluster_range=range(1, 7),
                                        view=(60, 40),
                                        output=output_path)
NUM_CLUSTERS = 2
agglo = cluster.AgglomerativeClustering(n_clusters=NUM_CLUSTERS)
agglo.fit(vectors_3d_slice)  # clustering
labels = agglo.labels_
main_df_slice['cluster_agglomerative'] = labels
plot_silhouette(scores_agglomerative,
                output="word2vec_agglomerative_silhouette-scores.png")
print("Agglomerative clustering completed (GloVe)")

# J.M.W. TURNER
turner_indices = main_df[main_df["artistId"] == 558].index.tolist()
other_indices = main_df.index.tolist()
# labels: green (#AEF498) for titles by Turner (others black)
turner_labels = ["#AEF498" if index in turner_indices
                 else "black" for index in other_indices]
scatterplot3D(vectors_3d,
              color=turner_labels,
              view=(60, 40),
              output="word2vec_turner.png")
# save data frames
main_df.to_csv(df_f+"main_df_clustered.csv", index=False)
main_df_slice.to_csv(df_f+"main_df_slice_clustered.csv", index=False)
print("main_df_clustered.csv created")
print("main_df_slice_clustered.csv created")

# GLOVE
# GloVe to Word2Vec
glove_file = datapath(glove_path)
word2vec_glove_file = get_tmpfile("glove.6B.300d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)
M_glove = KeyedVectors.load_word2vec_format(word2vec_glove_file)
M_glove_3d = pca_3d.fit_transform(M_glove.vectors)  # 3-D PCA of full model

main_df["in_glove"] = main_df["title_sw"].apply(lambda x:
                                                title_filter(x,
                                                             M_glove)).to_list()
glove_df = main_df[main_df["in_glove"] == True]
valid_titles = glove_df["title_sw"].to_list()

# sentence vectors
X_glove = []
for title in valid_titles:
    X_glove.append(sent_vectorizer(title, M_glove))
X_glove_3d = pca_3d.fit_transform(X_glove)  # 300d to 3d

# vectors to data frame
glove_df = main_df[main_df['in_glove'] == True].reset_index()
glove_df[['feature1',
          'feature2',
          'feature3']] = pd.DataFrame(X_glove_3d)
glove_vectors_3d = np.array([feature for feature in
                             glove_df[['feature1',
                                       'feature2',
                                       'feature3']].values])
scatterplot3D(glove_vectors_3d,
              view=(60, 40),
              output="glove_titles.png")

# K-MEANS: NLTK (cos)
output_path = "glove_nltk-cos_{}-clusters.png"  # format
glove_scores_nltk = SilhouetteScores(glove_vectors_3d,
                                     mode="nltk",
                                     cluster_range=range(1, 7),
                                     view=(60, 40),
                                     output=output_path)
plot_elbow(glove_scores_nltk[0],
           output="glove_nltk-cos_elbow-method.png")
plot_silhouette(glove_scores_nltk[1],
                output="glove_nltk-cos_silhouette-scores.png")
# cosine_distance
NUM_CLUSTERS = 3
kclusterer = KMeansClusterer(NUM_CLUSTERS,
                             distance=nltk.cluster.util.cosine_distance,
                             repeats=25, rng=rng)
# cluster
glove_df['cluster_kmeans'] = kclusterer.cluster(glove_vectors_3d,
                                                assign_clusters=True)
glove_df['centroid_kmeans'] = glove_df['cluster_kmeans'].apply(lambda x:
                                                               kclusterer.means()[x])
print("k-means clustering (NLTK) completed (GloVe)")

# K-MEANS: SCI-KIT LEARN (euc)
output_path = "glove_sklearn-euc_{}-clusters.png"  # format
glove_scores_sklearn = SilhouetteScores(glove_vectors_3d,
                                        mode="sklearn",
                                        cluster_range=range(1, 7),
                                        view=(60, 40),
                                        output=output_path)
plot_elbow(glove_scores_sklearn[0],
           output="glove_sklearn-euc_elbow-method.png")
plot_silhouette(glove_scores_sklearn[1],
                output="glove_sklearn-euc_silhouette-scores.png")
print("k-means clustering (sklearn) completed (GloVe)")

# DBSCAN
glove_df["is_duplicate"] = glove_df["title_sw"].apply(lambda x:
                                                      str(x)).duplicated()
duplicates_df = glove_df[glove_df["is_duplicate"] == False]
glove_vectors_3d_clean = np.array([feature for feature in
                                   duplicates_df[["feature1",
                                                  "feature2",
                                                  "feature3"]].values])
# cluster
dbs = cluster.DBSCAN(eps=0.21, min_samples=129).fit(glove_vectors_3d_clean)
labels = dbs.labels_
glove_df.loc[glove_df['is_duplicate'] == False, 'cluster_dbscan'] = labels
scatterplot3D(glove_vectors_3d_clean,
              color=labels,
              view=(60, 40),
              output="glove_dbscan_cluster.png")
print("DBSCAN clustering completed (GloVe)")

# AGGLOMERATIVE CLUSTERING
glove_df_slice = glove_df[glove_df['is_duplicate']
                          == False].sample(n=20000, random_state=42)
glove_vectors_3d_slice = np.array([feature for feature in
                                   glove_df_slice[["feature1",
                                                   "feature2",
                                                   "feature3"]].values])
output_path = "glove_agglomerative_{}-clusters.png"
scores_agglomerative = SilhouetteScores(glove_vectors_3d_slice,
                                        mode="agglomerative",
                                        cluster_range=range(1, 7),
                                        view=(60, 40),
                                        output=output_path)
NUM_CLUSTERS = 3
agglo = cluster.AgglomerativeClustering(n_clusters=NUM_CLUSTERS)
agglo.fit(glove_vectors_3d_slice)  # cluster
labels = agglo.labels_
glove_df_slice['cluster_agglomerative'] = labels
plot_silhouette(scores_agglomerative,
                output="glove_agglomerative_silhouette-scores.png")
print("Agglomerative clustering completed (GloVe)")

# J.M.W. TURNER
turner_indices = glove_df[glove_df["artistId"] == 558].index.tolist()
glove_indices = glove_df.index.tolist()
# labels: green (#AEF498) for titles by Turner (others black)
turner_labels = ["#AEF498" if index in turner_indices
                 else "black" for index in glove_indices]
scatterplot3D(glove_vectors_3d,
              color=turner_labels,
              view=(60, 40),
              output="glove_turner.png")
# save data frames
glove_df.to_csv(df_f+"glove_df_clustered.csv", index=False)
glove_df_slice.to_csv(df_f+"glove_df_slice_clustered.csv", index=False)
print("glove_df_clustered.csv created")
print("glove_df_slice_clustered.csv created")


# if __name__ == '__main__':
#    # code below is only executed when the module is run directly
