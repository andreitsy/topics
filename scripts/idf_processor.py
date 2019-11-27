import pandas as pd
import os
import itertools
import csv
from com.expleague.media_space.topics.file_read_util import FileReadUtil
from sklearn.feature_extraction.text import TfidfVectorizer
from com.expleague.media_space.topics.embedding_model import GasparettiTextNormalizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    texts = pd.read_csv(os.path.join(DATA_DIR, "parsed-texts-dragnet.csv"), chunksize=1000)
    normilizer = GasparettiTextNormalizer()
    documents = list()

    for chunk in texts:
        for index, row in chunk.iterrows():
            text = row["text"]
            if isinstance(text, str):
                words_in_doc = itertools.chain(*normilizer.normalized_sentences(text))
                documents.append(" ".join(words_in_doc))

    vectorizer = TfidfVectorizer(stop_words=None, norm='l2', token_pattern=r"(?u)\b[^\s]+\b")
    vectorizer.fit_transform(documents)
    word_idf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    with open(os.path.join(DATA_DIR, 'idf_dragnet.txt'), 'w+') as fin:
        _, words = FileReadUtil.load_fasttext(os.path.join(DATA_DIR, "news_dragnet.vec"), limit=int(10e18))
        fin.write(str(len(words)) + '\n')
        for word in words:
            idf = word_idf[word]
            fin.write(str(idf))
            fin.write('\n')


if __name__ == "__main__":
    main()
