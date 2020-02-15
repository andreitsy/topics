import pandas as pd
import os
import itertools
import csv
import numpy as np
from collections import Counter
from com.expleague.media_space.topics.file_read_util import FileReadUtil
from sklearn.feature_extraction.text import TfidfVectorizer
from com.expleague.media_space.topics.embedding_model import GasparettiTextNormalizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    texts = pd.read_csv(os.path.join(DATA_DIR, "texts-news.csv"), chunksize=1000)
    normilizer = GasparettiTextNormalizer()
    counter_docs_with_word = Counter()
    total_num_of_docs = 0
    for chunk in texts:
        for index, row in chunk.iterrows():
            total_num_of_docs += 1
            text = row["text"]
            if isinstance(text, str):
                words_in_doc = set(itertools.chain(*normilizer.normalized_sentences(text)))
                for word in words_in_doc:
                    counter_docs_with_word[word] += 1

    with open(os.path.join(DATA_DIR, 'idf_dragnet.txt'), 'w+') as fin:
        _, words = FileReadUtil.load_fasttext(os.path.join(DATA_DIR, "news_dragnet.vec"))
        fin.write(str(len(words)) + '\n')
        for word in words:
            c = counter_docs_with_word[word]
            if c == 0:
                idf = 0
            else:
                idf = np.log(total_num_of_docs / c)
            fin.write(str(idf))
            fin.write('\n')


if __name__ == "__main__":
    main()
