import pandas as pd
import numpy as np
import os
import requests
import re

DATA_DIR = "/home/tsypia/git/topics/topic_modeling/models/gasparetti"


if __name__ == "__main__":

    with open(os.path.join(DATA_DIR, 'cluster_centroids.txt'), 'r') as f:
        num_centroids = f.readline()
        cluster_centroids = f.read().splitlines()

    with open(os.path.join(DATA_DIR, 'cluster_names.txt'), 'r') as f:
        num_names = f.readline()
        cluster_names = f.read().splitlines()
    assert len(cluster_centroids) == len(cluster_names)
    nums = len([x for x in cluster_names if x != '-ignore-'])
    with open(os.path.join(DATA_DIR, 'cluster_names_filtered.txt'), 'w+') as f_n:
        with open(os.path.join(DATA_DIR, 'cluster_centroids_filtered.txt'), 'w+') as f_c:
            f_n.write(str(nums) + '\n')
            f_c.write(str(nums) + ' 100\n')
            for i in range(len(cluster_names)):
                if cluster_names[i] != '-ignore-':
                    f_n.write(cluster_names[i] + '\n')
                    f_c.write(cluster_centroids[i] + '\n')
