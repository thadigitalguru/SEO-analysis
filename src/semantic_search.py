"""
Build a tiny semantic search index over SERP results or research notes using MiniLM embeddings.
"""
import argparse, json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

def build_index(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    nn = NearestNeighbors(n_neighbors=min(5, len(texts)), metric="cosine")
    nn.fit(emb)
    return model, nn, emb

def query_index(model, nn, emb, texts, q, top_k=5):
    q_emb = model.encode([q], normalize_embeddings=True)
    distances, indices = nn.kneighbors(q_emb, n_neighbors=min(top_k, len(texts)))
    results = []
    for d, i in zip(distances[0], indices[0]):
        results.append({"text": texts[i], "distance": float(d)})
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/examples/serp_samples.csv")
    parser.add_argument("--query", default="marathon shoe tips for beginners")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    corpus = (df["title"] + ". " + df["snippet"]).tolist()
    model, nn, emb = build_index(corpus)
    res = query_index(model, nn, emb, corpus, args.query, top_k=3)
    print(json.dumps(res, indent=2))