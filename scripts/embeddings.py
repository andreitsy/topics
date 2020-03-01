import io
import os
import numpy as np
from sklearn.preprocessing import normalize
from itertools import combinations
from scipy.spatial import distance_matrix

from com.expleague.media_space.topics.fast_qt import FastQt

DATA_DIR = "/home/tsypia/git/topics/topic_modeling/models/gasparetti"


def load_fasttext(fname, skip=0, limit=10e9):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = [int(x) for x in fin.readline().split()]
    n = int(min(n, limit))
    data = np.zeros((n, d), dtype=np.float32)
    words = np.empty(n, dtype=object)

    i = 0
    for line in fin:
        if i >= skip:
            tokens = line.rstrip().split(' ')
            data[i - skip] = [np.float32(x) for x in tokens[1:]]
            words[i - skip] = tokens[0]

        i += 1
        if (i - skip) == limit:
            break
    fin.close()
    return data, words


# noinspection PyArgumentList
def calculate_clustering(data, words, threshold=0.8, min_cluster=100):
    cluster_words = list()

    def callback_cluster(indices, dist):
        centroid = np.mean(data[indices], axis=0)
        cluster_words.append([words[indices], centroid])

    # labels = FastQt(threshold, min_cluster).fit(data, lambda indices, dist: print(words[indices][:10000]))
    labels = FastQt(threshold, min_cluster).fit(data, callback_cluster)
    without_clusters = (labels == -1).sum()
    # print('Without cluster:', without_cluster_num)
    # print('Number clusters:', len(cluster_words))
    centroids = list()
    for i, cluster in enumerate(cluster_words):
        centroids.append(cluster[1])
    dist_matrix = distance_matrix(centroids, centroids)
    inverse_dist_sum = 0
    for i, j in combinations(range(len(cluster_words)), 2):
        inverse_dist_sum += 1 / dist_matrix[i][j]
    return inverse_dist_sum, without_clusters, cluster_words


def save_clustering(cluster_words, data, threshold, min_cluster):
    cluster_names_file = f"cluster_names_{'{:.2f}'.format(threshold)}_{min_cluster}.txt"
    cluster_words_file = f"cluster_words_{'{:.2f}'.format(threshold)}_{min_cluster}.txt"
    cluster_centroids_file = f"cluster_centroids_{'{:.2f}'.format(threshold)}_{min_cluster}.txt"
    with open(os.path.join(DATA_DIR, cluster_names_file), 'w+') as f_names:
        with open(os.path.join(DATA_DIR, cluster_words_file), 'w+') as f_words:
            with open(os.path.join(DATA_DIR, cluster_centroids_file), 'w+') as f_centroids:
                f_names.write(str(len(cluster_words)) + '\n')
                f_words.write(str(len(cluster_words)) + '\n')
                f_centroids.write(str(len(cluster_words)) + ' ' + str(data.shape[1]) + '\n')
                for i, cluster in enumerate(cluster_words):
                    f_names.write(cluster[0][0] + "|" + str(len(cluster[0])) + '\n')
                    f_words.write(" ".join(cluster[0]) + '\n')
                    f_centroids.write(" ".join([str(x) for x in cluster[1]]) + '\n')


if __name__ == "__main__":
    data, words = load_fasttext(os.path.join(DATA_DIR, "news_dragnet.vec"), 200, 100000)
    data = normalize(data)
    min_cluster = 50
    for th in range(58, 64, 2):
        threshold = th / 100.0
        inverse_dist, without_cluster_num, cluster_words = \
            calculate_clustering(data, words, threshold=threshold, min_cluster=min_cluster)
        save_clustering(cluster_words, data, threshold, min_cluster)
        print(threshold, inverse_dist, without_cluster_num, len(cluster_words))
