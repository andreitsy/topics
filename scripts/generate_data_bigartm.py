import pandas as pd
import numpy as np
import os
from com.expleague.media_space.topics.embedding_model import GasparettiTextNormalizer
import itertools

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

if __name__ == "__main__":
    texts = pd.read_csv(os.path.join(DATA_DIR, "gasparetti_small.csv"), chunksize=1000)
    normilizer = GasparettiTextNormalizer()
    documents = list()
    if not os.path.exists(os.path.join(DATA_DIR, "bigartm", "gasparetti.txt")):
        with open(os.path.join(DATA_DIR, "bigartm", "gasparetti.txt"), "w+") as f:
            for chunk in texts:
                for index, row in chunk.iterrows():
                    text = normilizer.normalized_sentences(row["text"])
                    url = row["url"]
                    words_in_doc = itertools.chain(*text)
                    f.write(url + " " + " ".join(words_in_doc) + "\n")
