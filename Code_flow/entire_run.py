"""
This is tentative code and needs to be cleaned out properly. We need to decide what model are we runnign finally, only sentiment or with clusters since clusters reduced the score
"""

from sentiment_function import load_finbert, compute_daily_sentiment
from cluster_function import *

# get finbert score
news_for_day = []
tokenizer, model = load_finbert()
sentiment_result = compute_daily_sentiment(news_for_day, tokenizer, model)

# load and clean data
gold_df = pd.read_csv("gold-dataset-sinha-khandait.csv")
headlines = gold_df["News"].dropna().astype(str).tolist()
cleaned_headlines = [clean_text(h) for h in headlines]

# train model
topic_model = setup_topic_model()
topics, probs = topic_model.fit_transform(cleaned_headlines)

# HDBSCAN
prob_matrix = np.array(all_points_membership_vectors(topic_model.hdbscan_model))

# macro groups
topic_group_map = create_macro_groups(topic_model, prob_matrix)

# macro probabilities
log_macro_group_prob = compute_macro_probabilities(prob_matrix, topic_group_map)

# prediction
topic_id, macro_group = predict_macro_group(news_for_day, topic_model, topic_group_map)
print(f"Predicted Topic ID: {topic_id}, Macro Group: {macro_group}")
