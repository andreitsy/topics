import artm
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def compute_number_of_topics(limit):
    """
    Number of topics
    :return:
    """
    texts = pd.read_csv(os.path.join(DATA_DIR, "gasparetti_full.csv"), chunksize=10000)
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


def save_clusters_predicted(theta, url_list, output_file):
    df = pd.DataFrame(columns=['url', 'story_id_predicted'])
    for i in range(len(url_list)):
        df.loc[i] = [url_list[i], np.argmax(np.array(theta[i]))]
    df.to_csv(os.path.join(DATA_DIR, "bigartm", output_file), index=False)


if __name__ == "__main__":
    clusters, url_list = compute_number_of_topics(None)
    num_topics = len(set(clusters))
    lc = artm.messages.ConfigureLoggingArgs()
    lc.log_dir = os.path.join(os.path.join(DATA_DIR, "bigartm"))
    lib = artm.wrapper.LibArtm(logging_config=lc)

    batch_vectorizer = artm.BatchVectorizer(data_path=os.path.join(DATA_DIR, "bigartm", "gasparetti.txt"),
                                            data_format='vowpal_wabbit',
                                            target_folder=os.path.join(DATA_DIR, "bigartm", "gasparetti_batches"))
    dictionary = batch_vectorizer.dictionary
    # batch_vectorizer = artm.BatchVectorizer(data_path=os.path.join(DATA_DIR, "bigartm", "gasparetti_batches"),
    #                                         data_format='batches')
    # dictionary = artm.Dictionary()
    # dictionary.gather(data_path=os.path.join(DATA_DIR, "bigartm", "gasparetti_batches"))

    lda_model = artm.LDA(num_topics=num_topics, alpha=0.01, beta=0.001, cache_theta=True,
                         num_document_passes=5, dictionary=dictionary)
    lda_model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
    # top_tokens = lda_model.get_top_tokens(num_tokens=20)
    # print("LDA:")
    # for i, token_list in enumerate(top_tokens):
    #     print('Topic #{0}: {1}'.format(i, token_list[10:]))

    theta_lda = lda_model.get_theta()
    save_clusters_predicted(theta_lda, url_list, "lda_predicted.csv")

    artm_model = artm.ARTM(num_topics=num_topics,
                           num_document_passes=5,
                           dictionary=dictionary,
                           scores=[artm.PerplexityScore(name='s1')],
                           regularizers=[artm.SmoothSparseThetaRegularizer(name='r1', tau=-0.15)])
    artm_model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)
    theta_bigartm = artm_model.get_theta()
    save_clusters_predicted(theta_bigartm, url_list, "bigartm_predicted.csv")
