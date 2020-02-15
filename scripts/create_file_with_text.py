import pandas as pd
import numpy as np
import os
import requests
import re
from com.expleague.media_space.topics.embedding_model import GasparettiTextNormalizer
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

if __name__ == "__main__":
    texts = pd.read_csv(os.path.join(DATA_DIR, "texts-news.csv"), chunksize=1000)
    normilizer = GasparettiTextNormalizer()
    with open(os.path.join(DATA_DIR, 'data.txt'), 'w+') as the_file:
        for chunk in texts:
            for index, row in chunk.iterrows():
                text = row["text"]
                if isinstance(text, str):
                    for words in normilizer.normalized_sentences(text):
                        the_file.write(" ".join(words) + " ")
            the_file.flush()
