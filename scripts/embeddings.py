import io
import os
import numpy as np
from sklearn.preprocessing import normalize

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
def main():
    data, words = load_fasttext(os.path.join(DATA_DIR, "news_dragnet.vec"), 2000, 100000)
    # data, words = load_glove('lenta-decomp-new', 20000)

    data = normalize(data)
    #threshold = 0.8
    #min_cluster = 50
    threshold = 0.82
    min_cluster = 80

    cluster_words = list()

    def callback_cluster(indices, dist):
        centroid = np.mean(data[indices], axis=0)
        cluster_words.append([words[indices], centroid])

    # labels = FastQt(threshold, min_cluster).fit(data, lambda indices, dist: print(words[indices][:10000]))
    labels = FastQt(threshold, min_cluster).fit(data, callback_cluster)
    print('Without cluster:', (labels == -1).sum())
    print('Number clusters:', len(cluster_words))
    with open(os.path.join(DATA_DIR, "cluster_names.txt"), 'w+') as f_names:
        with open(os.path.join(DATA_DIR, "cluster_words.txt"), 'w+') as f_words:
            with open(os.path.join(DATA_DIR, "cluster_centroids_words.txt"), 'w+') as f_centroids:
                f_names.write(str(len(cluster_words)) + '\n')
                f_words.write(str(len(cluster_words)) + '\n')
                f_centroids.write(str(len(cluster_words)) + ' ' + str(data.shape[1]) + '\n')
                for i, cluster in enumerate(cluster_words):
                    print(f"{i} = {len(cluster[0])}")
                    f_names.write(cluster[0][0] + "|" + str(len(cluster[0])) + '\n')
                    f_words.write(" ".join(cluster[0]) + '\n')
                    f_centroids.write(" ".join([str(x) for x in cluster[1]]) + '\n')


if __name__ == "__main__":
    main()
