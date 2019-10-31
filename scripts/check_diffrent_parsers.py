from abc import abstractmethod
from dragnet import extract_content
from goose3 import Goose
from goose3.text import StopWords
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
            self._csv_writer = csv.DictWriter(self._outfile, fieldnames=["url", "text"])
            self._csv_writer.writeheader()
        return self._csv_writer

    def save_to_file_parsed_line(self, page_html: str, url: str):
        parse_res = self.parse_news_text(url, page_html)
        self.writer.writerow(parse_res)


class ParserNewsPaper(Parser):
    def parse_news_text(self, page_html: str, url: str) -> dict:
        article = Article("", language="en")
        article.set_html(page_html)
        article.parse()
        news_text = re.sub(r'\s+', r' ', article.text)
        return {'url': url, 'text': news_text}


class ParserDragnet(Parser):
    def parse_news_text(self, page_html: str, url: str) -> dict:
        news_text = re.sub(r'\s+', r' ', extract_content(page_html))
        return {'url': url, 'text': news_text}


class ParserGoose(Parser):
    def parse_news_text(self, page_html: str, url: str) -> dict:
        extractor = Goose({'target_language': 'en',
                           'strict': True,
                           'stopwords_class': StopWords})
        article = extractor.extract(raw_html=page_html)
        news_text = article.cleaned_text
        return {'url': url, 'text': news_text}


def main():
    parser = argparse.ArgumentParser(description="Downloads news from CSV")
    parser.add_argument("--outfile-goose", default="parsed-texts-goose.csv", help="name of result file goose3")
    parser.add_argument("--outfile-newspaper", default="parsed-texts-newspaper.csv", help="name of result file np")
    parser.add_argument("--outfile-dragnet", default="parsed-texts-dragnet.csv", help="name of result file dragnet")
    parser.add_argument("--in-file", default="saved-html-small.csv", help="input file")
    args = parser.parse_args()

    out_goose = os.path.join("resources", args.outfile_goose)
    out_dragnet = os.path.join("resources", args.outfile_dragnet)
    out_newspaper = os.path.join("resources", args.outfile_newspaper)
    in_file = os.path.join("resources", args.in_file)

    in_data = pd.read_csv(in_file, sep=',', iterator=True, chunksize=2)
    try:
        parsers = (ParserDragnet(out_newspaper),
                   ParserGoose(out_goose),
                   ParserNewsPaper(out_newspaper)
                   )

        for chunk in in_data:
            for index, row in chunk.iterrows():
                for parser in parsers:
                    parser.save_to_file_parsed_line(row['html'], row['url'])

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, exiting...")


if __name__ == "__main__":
    main()
