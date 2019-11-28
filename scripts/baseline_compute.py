import artm
import os
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import f1_score

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def compute_number_of_topics(limit):
    """
    Number of topics
    :return:
    """
    texts = pd.read_csv(os.path.join(DATA_DIR, "gasparetti_small.csv"), chunksize=10000)
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


class ScoreComputer:
    def __init__(self, story_ids):
        self.story_ids = story_ids

    def compute_score(self, predicted_stories_list):
        return adjusted_rand_score(self.story_ids, predicted_stories_list)


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
    print("num_topics={}, alpha={}, beta={}, "
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
    print("num_topics={}, tau={},"
          "bigARTM score = {}".format(num_topics, tau, score))


if __name__ == "__main__":
    clusters, url_list = compute_number_of_topics(None)
    score_computer = ScoreComputer(clusters)
    num_topics = len(set(clusters))
    print("Number of topics initially: ", num_topics)
    lc = artm.messages.ConfigureLoggingArgs()
    lc.log_dir = os.path.join(os.path.join(DATA_DIR, "bigartm"))
    lib = artm.wrapper.LibArtm(logging_config=lc)

    # batch_vectorizer = artm.BatchVectorizer(data_path=os.path.join(DATA_DIR, "bigartm", "gasparetti.txt"),
    #                                         data_format='vowpal_wabbit',
    #                                         target_folder=os.path.join(DATA_DIR, "bigartm", "gasparetti_batches"))
    # dictionary = batch_vectorizer.dictionary

    batch_vectorizer = artm.BatchVectorizer(data_path=os.path.join(DATA_DIR, "bigartm", "gasparetti_batches"),
                                            data_format='batches')
    dictionary = artm.Dictionary()
    dictionary.gather(data_path=os.path.join(DATA_DIR, "bigartm", "gasparetti_batches"))
    parameters_variate_lda([0.035],
                           [0.039],
                           [num_topics - 50, num_topics],
                           dictionary, batch_vectorizer, score_computer)

    # df.to_csv(os.path.join(DATA_DIR, "bigartm", "lda_predicted.csv"), index=False)
    parameters_variate_big_artm([num_topics - 50, num_topics], [-0.17], dictionary, batch_vectorizer,
                                score_computer)
