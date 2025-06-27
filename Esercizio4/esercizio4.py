import pandas as pd
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan
import matplotlib.pyplot as plt
from bertopic import BERTopic
from tqdm import tqdm
import os
os.makedirs("visualizza", exist_ok=True)


# Setup
nltk.download("punkt")
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv("tripadvisor_hotel_reviews.csv", sep=",")
if df.columns[0] == 'Review,Rating':
    df[['Review', 'Rating']] = df['Review,Rating'].str.extract(r'^(.*),(\d+)$')
    df.drop(columns=['Review,Rating'], inplace=True)

# Preprocessing with progress bar
def preprocessa_testo(frase):
    tokens = word_tokenize(str(frase), language='english')
    tokens = [token.lower() for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words("english"))
    tokens_senza_stopwords = [token for token in tokens if token not in stop_words]
    doc = nlp(" ".join(tokens_senza_stopwords))
    return " ".join([token.lemma_ for token in doc])

tqdm.pandas(desc="Preprocessing reviews")
df["Review_preprocessed"] = df["Review"].progress_apply(preprocessa_testo)

# Embedding
model = SentenceTransformer("thenlper/gte-small")
embeddings = model.encode(df["Review_preprocessed"].tolist(), show_progress_bar=True)

# UMAP (5D) with progress
print("Applying UMAP (5D) for clustering...")
reducer = umap.UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42, verbose=True)
embeddings_umap = reducer.fit_transform(embeddings)

# HDBSCAN clustering
print("Clustering with HDBSCAN...")
hdbscan_model = hdbscan.HDBSCAN(
    min_cluster_size=50,
    metric="euclidean",
    cluster_selection_method="eom"
)
cluster_labels = hdbscan_model.fit_predict(embeddings_umap)
df["cluster"] = cluster_labels

# UMAP (2D) for visualization
print("Applying UMAP (2D) for visualization...")
umap_2d = umap.UMAP(n_components=5, min_dist=0.0, metric="cosine", random_state=42)
embedding_2d = umap_2d.fit_transform(embeddings)

# Plot
plt.figure(figsize=(12, 8))
palette = plt.cm.get_cmap('tab10', len(set(cluster_labels)))
colors = [palette(label) if label != -1 else (0.6, 0.6, 0.6) for label in cluster_labels]
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=10, alpha=0.8)
plt.title("Visualizzazione Clustering con HDBSCAN (Outlier in grigio)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("clustering_hdbscan.png", dpi=300)
plt.show()

# BERTopic model
print("Fitting BERTopic model...")
topic_model = BERTopic(
    embedding_model=model,
    umap_model=reducer, #modello hdbsca instanziato prima
    hdbscan_model=hdbscan_model,
    verbose=True
)
topics, probs = topic_model.fit_transform(df["Review_preprocessed"].tolist(), embeddings)

# Output
print(topic_model.get_topic_info())
topic_model.visualize_documents(df["Review_preprocessed"].tolist(), topics).write_html("visualizza/bertopic_docs.html")
topic_model.visualize_barchart().write_html("visualizza/bertopic_barchart.html")
topic_model.visualize_hierarchy().write_html("visualizza/bertopic_hierarchy.html")
topic_model.visualize_heatmap().write_html("visualizza/bertopic_heatmap.html")

# Ottieni le informazioni sui topic
topics_info = topic_model.get_topic_info()

# Ottieni le parole chiave per ciascun topic
topic_keywords = {}
for topic_id in topics_info["Topic"]:
    if topic_id == -1:
        continue  # Salta gli outlier
    words = topic_model.get_topic(topic_id)
    topic_keywords[topic_id] = [{"word": w, "score": s} for w, s in words]

# Salva in JSON
import json
with open("topic_keywords.json", "w", encoding="utf-8") as f:
    json.dump(topic_keywords, f, indent=2, ensure_ascii=False)

# Salva anche in formato CSV (opzionale)
import csv
with open("topic_keywords.csv", "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Topic", "Word", "Score"])
    for topic_id, words in topic_keywords.items():
        for item in words:
            writer.writerow([topic_id, item["word"], item["score"]])