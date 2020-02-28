import pandas as pd
import numpy as np
import os
from com.expleague.media_space.topics.embedding_model import GasparettiTextNormalizer
import itertools

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


if __name__ == "__main__":
    texts = pd.read_csv(os.path.join(DATA_DIR, "parsed-texts-dragnet.csv"))
    normilizer = GasparettiTextNormalizer()
    output_df = pd.DataFrame(columns=texts.columns)

    for index, row in texts.iterrows():
        text = row["text"]
        title = row["title"]
        timevar = row["timestamp"]
        if isinstance(text, str) and timevar.isdigit():
            if "We are sorry, you need to be a subscriber" in text:
                continue
            if "Please purchase a subscription" in text:
                continue
            sentence_texts = set(itertools.chain.from_iterable(normilizer.normalized_sentences(text)))
            sentence_title = set(itertools.chain.from_iterable(normilizer.normalized_sentences(title)))
            if (len(sentence_texts) < 100) or (len(sentence_texts.intersection(sentence_title)) == 0):
                continue
        else:
            continue
        output_df.loc[index] = row
    output_df.to_csv(os.path.join(DATA_DIR, "texts-news.csv"), index=False)
