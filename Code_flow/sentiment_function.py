import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.nn.functional import softmax


def load_finbert(model_name="ProsusAI/finbert"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


def compute_daily_sentiment(news_list, tokenizer, model):
    logits_list = []

    for sentence in news_list:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits.squeeze()
        logits_list.append(logits)

    if not logits_list:
        return {
            "Positive": 0.0,
            "Neutral": 0.0,
            "Negative": 0.0,
            "Final Sentiment": "neutral",
        }

    stacked_logits = torch.stack(logits_list)
    avg_logits = torch.mean(stacked_logits, dim=0)
    probabilities = softmax(avg_logits, dim=0)

    labels = model.config.id2label
    final_probs = {labels[i]: float(probabilities[i]) for i in range(len(labels))}
    final_sentiment = max(final_probs, key=final_probs.get)

    return {
        "Positive": final_probs.get("positive", 0.0),
        "Neutral": final_probs.get("neutral", 0.0),
        "Negative": final_probs.get("negative", 0.0),
        "Final Sentiment": final_sentiment,
    }
