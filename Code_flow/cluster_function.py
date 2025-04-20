import pandas as pd
import numpy as np
import re

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

from umap import UMAP
from hdbscan import HDBSCAN, all_points_membership_vectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score


def clean_text(text: str) -> str:
    months = r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b"
    directions = r"\b(up|down|higher|lower|rise|rises|fall|falls|gain|gains|loses|loss|rebound|slip|climb|surge|drop|drops|edged|edges|recover|recovery|recovers|flat)\b"
    numbers = r"[\d\.,]+[%$]?|\d{1,3}(,\d{3})*(\.\d+)?|\d+"
    symbols = r"\/oz|rs|bn|usd|\$|%|oz"

    text = text.lower()
    text = re.sub(months, "", text)
    text = re.sub(directions, "", text)
    text = re.sub(numbers, "", text)
    text = re.sub(symbols, "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def setup_topic_model():
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=10,
        max_df=0.5,
        max_features=5000,
        token_pattern=r"(?u)\b[\w\-]+\b",
    )
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=60,
        min_samples=10,
        cluster_selection_epsilon=0.1,
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=hdbscan_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer,
        verbose=True,
    )
    return topic_model


def create_macro_groups(topic_model, prob_matrix):
    from scipy.cluster.hierarchy import linkage, fcluster

    similarity = cosine_similarity(topic_model.topic_embeddings_)
    distance = 1 - similarity
    Z = linkage(distance, method="average")

    best_score = float("inf")
    best_t = None
    for t in range(10, 20, 2):
        try:
            cluster_labels = fcluster(Z, t=t, criterion="maxclust")
            db_score = davies_bouldin_score(distance, cluster_labels)
            if db_score < best_score:
                best_score = db_score
                best_t = t
        except:
            continue

    macro_labels = fcluster(Z, t=best_t, criterion="maxclust")
    topic_group_map = pd.DataFrame(
        {"Original_Topic": np.arange(len(macro_labels)), "Macro_Group": macro_labels}
    )
    return topic_group_map


def compute_macro_probabilities(prob_matrix, topic_group_map):
    macro_group_map_vector = (
        topic_group_map.set_index("Original_Topic")
        .loc[list(range(prob_matrix.shape[1]))]["Macro_Group"]
        .values
    )
    macro_group_onehot = pd.get_dummies(macro_group_map_vector)
    macro_group_prob = prob_matrix @ macro_group_onehot.values

    epsilon = 1e-12
    log_macro_group_prob = np.log(macro_group_prob + epsilon)
    return log_macro_group_prob


def predict_macro_group(test_sentence, topic_model, topic_group_map):
    cleaned = clean_text(test_sentence)
    topic, prob = topic_model.transform([cleaned])

    if topic[0] == -1:
        return "noise", -1
    else:
        macro_group = topic_group_map.loc[
            topic_group_map["Original_Topic"] == topic[0], "Macro_Group"
        ].values[0]
        return topic[0], macro_group
