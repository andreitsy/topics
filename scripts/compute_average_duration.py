import pandas as pd
import numpy as np
import os
from com.expleague.media_space.topics.embedding_model import GasparettiTextNormalizer
import itertools
from datetime import datetime, timedelta
from collections import defaultdict

DATA_DIR = "/mnt/c/Users/griff/YandexDisk/cs_center/topics/data"


def read_news_corpora():
    """
    Read news file newsCorpora.csv:
    Attribute Information:
        FILENAME #1: newsCorpora.csv
        DESCRIPTION: News pages
        FORMAT: ID TITLE URL PUBLISHER CATEGORY STORY HOSTNAME TIMESTAMP

    where:
        ID Numeric ID
        TITLE News title
        URL Url
        PUBLISHER Publisher name
        CATEGORY News category (b = business, t = science and technology, e = entertainment, m = health)
        STORY Alphanumeric ID of the cluster that includes news about the same story
        HOSTNAME Url hostname
        TIMESTAMP Approximate time the news was published,
                  as the number of milliseconds since the epoch 00:00:00 GMT, January 1, 1970
    :return: pd.dataframe with news info
    """
    header = ["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"]
    news_corpora = pd.read_csv(os.path.join(DATA_DIR, "newsCorpora.csv"), sep='\t', header=None,
                               names=header, low_memory=False,
                               dtype={"ID": int, "TITLE": str, "URL": str, "CATEGORY": str, "STORY": str,
                                      "HOSTNAME": str, "TIMESTAMP": int})
    # news_corpora.set_index('URL', inplace=True)
    return news_corpora


if __name__ == "__main__":
    texts = read_news_corpora()
    normilizer = GasparettiTextNormalizer()

    def convert_epoch(ts):
        return datetime.utcfromtimestamp(int(ts) / 1000)

    times = defaultdict(list)
    times_epoch = defaultdict(list)
    num_errors = 0
    for index, row in texts.iterrows():
        time_epoch = row["TIMESTAMP"]
        times_epoch[row["STORY"]].append(int(time_epoch))
        try:
            date = convert_epoch(time_epoch)
        except ValueError:
            num_errors += 1
            # print("Error:", row)
        else:
            story = row["STORY"]
            times[story].append(date)
    print("Num errors: ", num_errors)
    print("number_of_stories", len(times))
    print(list(times.items())[:10])
    durations = list()
    number_of_news = 0
    for k in times:
        number_of_news += len(times[k])
        if len(times[k]) > 1:
            story_news = sorted(times[k])
            duration = story_news[-1] - story_news[0]
            durations.append(duration.total_seconds())
    print("number_of_news = ", number_of_news)
    print(np.max(durations))
    print(np.mean(durations))
    print(np.median(durations))

    durations_ = list()
    for k in times_epoch:
        number_of_news += len(times_epoch[k])
        if len(times_epoch[k]) > 1:
            story_news = sorted(times_epoch[k])
            duration = story_news[-1] - story_news[0]
            durations_.append(duration)
    print(np.max(durations_))
    print(np.mean(durations_))
    print(np.median(durations_))

