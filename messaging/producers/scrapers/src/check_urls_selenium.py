import requests
import json

url = "https://www.reuters.com/pf/api/v3/content/fetch/articles-by-section-alias"
params = {
    "query": json.dumps({
        "offset": 0,
        "size": 25,
        "sectionAlias": "technology",
        "website": "reuters"
    })
}

headers = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.reuters.com/technology/"
}

response = requests.get(url, params=params, headers=headers)
data = response.json()

articles = []
for item in data.get("result", {}).get("articles", []):
    title = item.get("title")
    url = "https://www.reuters.com" + item.get("canonical_url", "")
    if title and url:
        articles.append((title.strip(), url))

print(f"[✓] {len(articles)} articles Reuters détectés :\n")
for title, url in articles[:10]:
    print("-", title)
    print(" ", url)
