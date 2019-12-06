#!/usr/bin/env python

import artm
import datetime
import os
import argparse
import pandas as pd
import numpy as np
import logging
import itertools
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import f1_score
from com.expleague.media_space.topics_script import TopicsScript
from com.expleague.media_space.input import NewsGasparettiInput
from com.expleague.media_space.topics.params import ProcessingParams, StartupParams
from com.expleague.media_space.topics.embedding_model import GasparettiTextNormalizer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
BIGARTM_DIR = os.path.join(DATA_DIR, "bigartm")


class ScoreComputer:
    def __init__(self, story_ids):
        self.story_ids = story_ids

    def compute_score(self, predicted_stories_list):
        return adjusted_rand_score(self.story_ids, predicted_stories_list)


def generate_data_for_bigartm(file_path):
    texts = pd.read_csv(file_path, chunksize=10000)
    normilizer = GasparettiTextNormalizer()
    if not os.path.exists(os.path.join(BIGARTM_DIR, "gasparetti.txt")):
        with open(os.path.join(BIGARTM_DIR, "gasparetti.txt"), "w+") as f:
            for chunk in texts:
                for index, row in chunk.iterrows():
                    text = normilizer.normalized_sentences(row["text"])
                    url = row["url"]
                    words_in_doc = itertools.chain(*text)
                    f.write(url + " " + " ".join(words_in_doc) + "\n")


def compute_number_of_topics(file_path, limit):
    """
    Number of topics
    :return:
    """
    texts = pd.read_csv(file_path, chunksize=10000)
    clusters = list()
    url_list = list()
    i = 0
    for chunk in texts:
        for index, row in chunk.iterrows():
            clusters.append(row["story"])
            url_list.append(row["url"])
            i += 1
            if limit and i >= limit:
                break
        else:
            # Continue if the inner loop wasn't broken.
            continue
            # Inner loop was broken, break the outer.
        break
    return clusters, url_list


def get_df_clusters_predicted(theta, url_list):
    df = pd.DataFrame(columns=['url', 'story_id_predicted'])
    for i in range(len(url_list)):
        df.loc[i] = [url_list[i], np.argmax(np.array(theta[i]))]
    return df


def parameters_variate_lda(alpha_range, beta_range, num_topics_range, dictionary, batch_vectorizer, score_computer):
    for alpha in alpha_range:
        for beta in beta_range:
            for num_topics in num_topics_range:
                compute_lda(num_topics, alpha, beta, dictionary, batch_vectorizer, score_computer)


def compute_lda(num_topics, alpha, beta, dictionary, batch_vectorizer, score_computer):
    lda_model = artm.LDA(num_topics=num_topics, alpha=alpha, beta=beta, cache_theta=True,
                         num_document_passes=5, dictionary=dictionary)
    lda_model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
    theta_lda = lda_model.get_theta()
    lda_predicts = get_df_clusters_predicted(theta_lda, url_list)
    score = score_computer.compute_score(lda_predicts["story_id_predicted"])
    logging.info("num_topics={}, alpha={}, beta={}, "
                 "LDA score = {}".format(num_topics, alpha, beta, score))


def parameters_variate_big_artm(num_topics_range, tau_range, dictionary, batch_vectorizer, score_computer):
    for num_topics in num_topics_range:
        for tau in tau_range:
            compute_big_artm(num_topics, tau, dictionary, batch_vectorizer, score_computer)


def compute_big_artm(num_topics, tau, dictionary, batch_vectorizer, score_computer):
    artm_model = artm.ARTM(num_topics=num_topics,
                           num_document_passes=5,
                           dictionary=dictionary,
                           scores=[artm.PerplexityScore(name='s1')],
                           regularizers=[artm.SmoothSparseThetaRegularizer(name='r1', tau=tau)], cache_theta=True)
    artm_model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
    theta_bigartm = artm_model.get_theta()
    bigartm_predicts = get_df_clusters_predicted(theta_bigartm, url_list)
    score = score_computer.compute_score(bigartm_predicts["story_id_predicted"])
    logging.info("num_topics={}, tau={},"
                 "bigARTM score = {}".format(num_topics, tau, score))


def compute_score_topic_modeling(score_cmp=None,
                                 min_sentence_len=6,
                                 topic_cos_threshold=0.7,
                                 news_clustering_threshold=0.025,
                                 news_clustering_min_cluster_size=4,
                                 stories_clustering_threshold=0.25,
                                 stories_clustering_min_cluster_size=2,
                                 ngrams_for_topics_labelling=3,
                                 stories_connecting_cos_threshold=0.6,
                                 story_window=4,
                                 lexic_result_word_num=10,
                                 sclale_dist=100,
                                 input_file_path="gasparetti_small.csv",
                                 models_path="/home/andreitsy/git/topics/data/models",
                                 start='10.03.2014',
                                 end='26.03.2014'):
    articles_input = NewsGasparettiInput(input_file_path)
    text_normalizer = GasparettiTextNormalizer()

    start = datetime.datetime.strptime(start, '%d.%m.%Y').replace(tzinfo=datetime.timezone.utc)
    end = datetime.datetime.strptime(end, '%d.%m.%Y').replace(tzinfo=datetime.timezone.utc)

    embedding_file_path = os.path.join(models_path, "news_dragnet.vec")
    idf_file_path = os.path.join(models_path, 'idf_dragnet.txt')
    cluster_centroids_file_path = os.path.join(models_path, 'cluster_centroids_filtered.txt')
    cluster_names_file_path = os.path.join(models_path, 'cluster_names_filtered.txt')
    topics_matching_file_path = os.path.join(models_path, 'topic_matching.txt')

    params_logging_str = f"FROM_DATE: {start}\n" \
        f"TO_DATE: {end}\n\n" \
        f"EMBEDDING_FILE_PATH: {embedding_file_path}\n" \
        f"IDF_FILE_PATH: {idf_file_path}\n" \
        f"CLUSTER_CENTROIDS_FILE_PATH: {cluster_centroids_file_path}\n\n" \
        f"MIN_SENTENCE_LEN: {min_sentence_len}\n" \
        f"TOPIC_COS_THRESHOLD: {topic_cos_threshold}\n" \
        f"NEWS_CLUSTERING_THRESHOLD: {news_clustering_threshold}\n" \
        f"NEWS_CLUSTERING_MIN_CLUSTER_SIZE: {news_clustering_min_cluster_size}\n" \
        f"STORIES_CLUSTERING_THRESHOLD: {stories_clustering_threshold}\n" \
        f"STORIES_CLUSTERING_MIN_CLUSTER_SIZE: {stories_clustering_min_cluster_size}\n" \
        f"NGRAMS_FOR_TOPICS_LABELLING: {ngrams_for_topics_labelling}\n" \
        f"STORIES_CONNECTING_COS_THRESHOLD: {stories_connecting_cos_threshold}\n" \
        f"STORY_WINDOW: {story_window}\n" \
        f"LEXIC_RESULT_WORD_NUM: {lexic_result_word_num}\n" \
        f"SCALE_DIST: {sclale_dist}\n"
    logging.info('Parameters used:\n' + params_logging_str)
    processor = TopicsScript(
        StartupParams(start, end),
        ProcessingParams(embedding_file_path, idf_file_path, cluster_centroids_file_path,
                         cluster_names_file_path, topics_matching_file_path, min_sentence_len,
                         topic_cos_threshold,
                         news_clustering_threshold,
                         news_clustering_min_cluster_size, stories_clustering_threshold,
                         stories_clustering_min_cluster_size, ngrams_for_topics_labelling,
                         stories_connecting_cos_threshold, story_window, lexic_result_word_num, sclale_dist))
    topic_news = processor.run(articles_input, text_normalizer, verbose=False)
    dict_clusters = dict()
    for cluster_id in topic_news:
        articles = topic_news[cluster_id]
        for article in articles:
            dict_clusters[article.id] = cluster_id

    output_clusters = pd.DataFrame(columns=["url", "timestamp", "story_id_predicted", "story_id"])
    for index, row in articles_input.df.iterrows():
        cluster_id = dict_clusters.get(row["url"], "0")
        output_clusters.loc[index] = [row["url"], row["timestamp"], cluster_id, row["story"]]
    if score_cmp:
        score = score_cmp.compute_score(output_clusters["story_id_predicted"])
        logging.info('TM score : ' + str(score) + "\n")


def parameters_topic_modeling(score_cmp, input_file_path,
                              topic_cos_threshold_range,
                              news_clustering_threshold_range,
                              news_clustering_min_cluster_size_range,
                              stories_connecting_cos_threshold_range,
                              min_sentence_len=4,
                              stories_clustering_threshold=0.28,
                              stories_clustering_min_cluster_size=2,
                              ngrams_for_topics_labelling=3,
                              story_window=4,
                              lexic_result_word_num=10,
                              sclale_dist=200):
    for topic_cos_threshold in topic_cos_threshold_range:
        for news_clustering_threshold in news_clustering_threshold_range:
            for news_clustering_min_cluster_size in news_clustering_min_cluster_size_range:
                for stories_connecting_cos_threshold in stories_connecting_cos_threshold_range:
                    compute_score_topic_modeling(score_cmp=score_cmp,
                                                 input_file_path=input_file_path,
                                                 min_sentence_len=min_sentence_len,
                                                 topic_cos_threshold=topic_cos_threshold,
                                                 news_clustering_threshold=news_clustering_threshold,
                                                 news_clustering_min_cluster_size=news_clustering_min_cluster_size,
                                                 stories_clustering_threshold=stories_clustering_threshold,
                                                 stories_clustering_min_cluster_size=stories_clustering_min_cluster_size,
                                                 ngrams_for_topics_labelling=ngrams_for_topics_labelling,
                                                 stories_connecting_cos_threshold=stories_connecting_cos_threshold,
                                                 story_window=story_window,
                                                 lexic_result_word_num=lexic_result_word_num,
                                                 sclale_dist=sclale_dist)


if __name__ == "__main__":
    time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    parser = argparse.ArgumentParser(description='Run topics matching')
    parser.add_argument('-i', '--input', type=str, default="gasparetti_small.csv",
                        help='Input news source')
    parser.add_argument('-l', '--log-file', type=str,
                        default=f"topics-script-log-{time_now}.txt",
                        help='Path to log file')
    args = parser.parse_args()

    input_file_path = os.path.join(DATA_DIR, args.input)
    generate_data_for_bigartm(input_file_path)

    logging.getLogger()
    logging.basicConfig(filename=args.log_file, filemode='w', level=logging.INFO)

    clusters, url_list = compute_number_of_topics(input_file_path, None)
    score_computer = ScoreComputer(clusters)
    num_topics = len(set(clusters))
    logging.info("Number of topics initially: " + str(num_topics))

    lc = artm.messages.ConfigureLoggingArgs()
    lc.log_dir = BIGARTM_DIR
    lib = artm.wrapper.LibArtm(logging_config=lc)

    if os.path.exists(os.path.join(os.path.join(BIGARTM_DIR, "gasparetti_batches"))):
        batch_vectorizer = artm.BatchVectorizer(data_path=os.path.join(BIGARTM_DIR, "gasparetti_batches"),
                                                data_format='batches')
        dictionary = artm.Dictionary()
        dictionary.gather(data_path=os.path.join(BIGARTM_DIR, "gasparetti_batches"))
    else:
        batch_vectorizer = artm.BatchVectorizer(data_path=os.path.join(BIGARTM_DIR, "gasparetti.txt"),
                                                data_format='vowpal_wabbit',
                                                target_folder=os.path.join(BIGARTM_DIR, "gasparetti_batches"))
        dictionary = batch_vectorizer.dictionary

    parameters_topic_modeling(score_computer, input_file_path,
                              [0.1, 0.3, 0.7],
                              [0.01, 0.05, 0.2],
                              [2, 5, 10],
                              [0.2, 0.5, 0.9])

    parameters_variate_lda([0.01, 0.05],
                           [0.02],
                           [num_topics],
                           dictionary, batch_vectorizer, score_computer)

    parameters_variate_big_artm([num_topics],
                                [-0.3, -0.27],
                                dictionary,
                                batch_vectorizer,
                                score_computer)
