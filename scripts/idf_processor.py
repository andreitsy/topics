import pandas as pd
import re
import os
import itertools
from com.expleague.media_space.topics.file_read_util import FileReadUtil
from sklearn.feature_extraction.text import TfidfVectorizer
from com.expleague.media_space.topics.embedding_model import SimpleTextNormalizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    texts = pd.read_csv(os.path.join(DATA_DIR, "parsed-texts-dragnet.csv"), chunksize=1000)
    normilizer = SimpleTextNormalizer()
    documents = list()

    for chunk in texts:
        for index, row in chunk.iterrows():
            text = row["text"]
            if isinstance(text, str):
                words_in_doc = itertools.chain(*normilizer.normalized_sentences(text))
                documents.append(" ".join(words_in_doc))

    tfidf = TfidfVectorizer(lowercase=True)  # , stop_words='english'
    tfidf_matrix = tfidf.fit_transform(documents)
    feature_names = tfidf.get_feature_names()
    word_idf = dict()
    for col in tfidf_matrix.nonzero()[1]:
        word_idf[feature_names[col]] = tfidf_matrix[0, col]
    print(word_idf.items()[0:100])
    assert 1 / 0
    with open(os.path.join(DATA_DIR, 'idf_dragnet.txt'), 'w+') as fin:
        _, words = FileReadUtil.load_fasttext(os.path.join(DATA_DIR, "news_dragnet.vec"))
        for word in words:
            idf = word_idf[word]
            fin.write(str(idf))
            fin.write('\n')


if __name__ == "__main__":
    main()
