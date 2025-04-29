import requests
import pandas as pd
import json
import time

base_url = "https://newsdata.io/api/1/news"
params = {
    # "apikey": key_value,
    "q": "gold metal economy",
    "language": "en",
    "category": "business,politics,world",
}

all_articles = []

while True:
    # API call
    response = requests.get(base_url, params=params)
    data = response.json()

    # Extract articles
    articles = data.get("results", [])
    all_articles.extend(articles)

    # Check if there is a next page
    next_page = data.get("nextPage", None)
    if not next_page:
        break
    else:
        params["page"] = next_page
        time.sleep(1)  # polite delay

# Now process all collected articles
records = []
for article in all_articles:
    pub_date = article.get("pubDate", "")[:10]  # Just YYYY-MM-DD
    title = article.get("title", "")
    if pub_date and title:  # Only if both exist
        records.append((pub_date, title))

# Create DataFrame
df = pd.DataFrame(records, columns=["Date", "Headline"])

# Drop duplicate headlines
df = df.drop_duplicates(subset=["Headline"]).reset_index(drop=True)

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

df.to_csv("fetched_gold_metal_economy_headlines.csv")
