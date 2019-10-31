"""

Attribute Information:

FILENAME #1: newsCorpora.csv (102.297.000 bytes)
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
TIMESTAMP Approximate time the news was published, as the number of milliseconds since the epoch 00:00:00 GMT, January 1, 1970


FILENAME #2: 2pageSessions.csv (3.049.986 bytes)
DESCRIPTION: 2-page sessions
FORMAT: STORY HOSTNAME CATEGORY URL

where:
STORY Alphanumeric ID of the cluster that includes news about the same story
HOSTNAME Url hostname
CATEGORY News category (b = business, t = science and technology, e = entertainment, m = health)
URL Two space-delimited urls representing a browsing session

"""

import pandas as pd
import os
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HEADER_CORPORA = ["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"]

if __name__ == "__main__":
    news_corpora = pd.read_csv(os.path.join(DATA_DIR, "newsCorpora.csv"), sep='\t', header=None, names=HEADER_CORPORA)
    url_check = dict(ID=list(), URL=list(), exist=list())
    url_avaiable = False

    for index, row in news_corpora.head().iterrows():
        url = row["URL"]
        id = row["ID"]
        request = requests.get(url, allow_redirects=True)
        if request.status_code == 200:
            url_avaiable = True
        else:
            url_avaiable = False
        print(url, "\t avaiable: ", url_avaiable)
