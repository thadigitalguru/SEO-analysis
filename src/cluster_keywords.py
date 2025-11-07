"""
Cluster keywords using sentence-transformer embeddings + KMeans.
Output: cluster id + representative keyword (closest to centroid).
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sentence_transformers import SentenceTransformer
from src.utils import read_keywords_csv, save_csv

def cluster_keywords(input_csv: str, output_csv: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", n_clusters: int = 5, random_state: int = 42):
    df = read_keywords_csv(input_csv)
    model = SentenceTransformer(model_name)
    texts = df["keyword"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(embeddings)
    df["cluster"] = labels

    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, embeddings)
    reps = {i: texts[idx] for i, idx in enumerate(closest)}

    df["cluster_rep"] = df["cluster"].map(reps)
    save_csv(df, output_csv)
    print("Cluster representatives:")
    for i in range(n_clusters):
        print(f"- {i}: {reps[i]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/examples/keywords.csv")
    parser.add_argument("--output", default="data/examples/keywords_clusters.csv")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--clusters", type=int, default=5)
    args = parser.parse_args()
    cluster_keywords(args.input, args.output, args.model, args.clusters)