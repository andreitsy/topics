from abc import abstractmethod
from dragnet import extract_content
from goose3 import Goose
from goose3.text import StopWords
from goose3.configuration import Configuration
from newspaper import Article
import argparse
import logging
import re
import pandas as pd
import csv
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s @ %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
)
logger = logging.getLogger(name='Parser')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class Parser:
    def __init__(self, outfile_name: str):
        self._outfile_name = outfile_name
        self._outfile = None
        self._csv_writer = None

    @abstractmethod
    def parse_news_text(self, page_html: str, url: str) -> dict:
        pass

    @property
    def writer(self):
        if self._csv_writer is None:
            self._outfile = open(self._outfile_name, "w", 1)
            self._csv_writer = csv.DictWriter(self._outfile, fieldnames=["url", "text", "title",
                                                                         "story", "category", "timestamp"])
            self._csv_writer.writeheader()
        return self._csv_writer

    def save_to_file_parsed_line(self, page_html: str, url: str, title: str, story: str, category: str, timestamp: str):
        parse_res = self.parse_news_text(page_html, url)
        parse_res.update(dict(title=title, story=story, timestamp=timestamp, category=category))
        self.writer.writerow(parse_res)


class ParserNewsPaper(Parser):
    _extractor = None

    def parse_news_text(self, page_html: str, url: str) -> dict:
        if self._extractor is None:
            self._extractor = Article("", language="en")
        self._extractor.set_html(page_html)
        self._extractor.parse()
        news_text = re.sub(r'\s+', r' ', self._extractor.text)
        return {'url': url, 'text': news_text}


class ParserDragnet(Parser):
    def parse_news_text(self, page_html: str, url: str) -> dict:
        news_text = re.sub(r'\s+', r' ', extract_content(page_html, encoding='utf-8'))
        return {'url': url, 'text': news_text}


class ParserGoose(Parser):
    _extractor = None

    def parse_news_text(self, page_html: str, url: str) -> dict:
        if self._extractor is None:
            config = Configuration()
            config.stopwords_class = StopWords
            extractor = Goose(config)
            self._extractor = extractor
        article = self._extractor.extract(raw_html=page_html)
        news_text = re.sub(r'\s+', r' ', article.cleaned_text)
        return {'url': url, 'text': news_text}


def read_news_corpora():
    header = ["ID", "TITLE", "URL", "PUBLISHER", "CATEGORY", "STORY", "HOSTNAME", "TIMESTAMP"]
    news_corpora = pd.read_csv(os.path.join(DATA_DIR, "newsCorpora.csv"), sep='\t', header=None,
                               names=header)
    news_corpora.set_index('URL', inplace=True)
    return news_corpora


def main():
    parser = argparse.ArgumentParser(description="Downloads news from CSV")
    parser.add_argument("--outfile-goose", default="parsed-texts-goose.csv", help="name of result file goose3")
    parser.add_argument("--outfile-newspaper", default="parsed-texts-newspaper.csv", help="name of result file np")
    parser.add_argument("--outfile-dragnet", default="parsed-texts-dragnet.csv", help="name of result file dragnet")
    parser.add_argument("--in-file", default="saved-html-small.csv", help="input file")
    args = parser.parse_args()

    out_goose = os.path.join(DATA_DIR, args.outfile_goose)
    out_dragnet = os.path.join(DATA_DIR, args.outfile_dragnet)
    out_newspaper = os.path.join(DATA_DIR, args.outfile_newspaper)
    in_file = os.path.join(DATA_DIR, args.in_file)

    in_data = pd.read_csv(in_file, sep=',', iterator=True, chunksize=100)

    news_corpora = read_news_corpora()
    try:
        parsers = (
            ParserDragnet(out_dragnet),
            # ParserGoose(out_goose),
            ParserNewsPaper(out_newspaper),
        )

        for chunk in in_data:
            for index, row in chunk.iterrows():
                for parser in parsers:
                    title = news_corpora.loc[row['url']]["TITLE"]
                    category = news_corpora.loc[row['url']]["CATEGORY"]
                    story = news_corpora.loc[row['url']]["STORY"]
                    timestamp = news_corpora.loc[row['url']]["TIMESTAMP"]
                    parser.save_to_file_parsed_line(row['html'], row['url'], title, story, category, timestamp)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, exiting...")


if __name__ == "__main__":
    main()
