"""
Embeddings-assisted internal linking suggestions.

Inputs: CSV with columns: url, title, content
Modes:
- Single target: suggest top-N candidates for a given target URL or topic text
- Batch: suggest top-N candidates for every page in the corpus (excluding self)

Outputs: CSV with columns: source_url, source_title, candidate_url, candidate_title, score, suggested_anchor
"""
import argparse
from typing import List, Optional, Tuple
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np


def load_pages(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"url", "title", "content"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input must contain columns: {sorted(required)}; missing: {sorted(missing)}")
    return df


def build_embeddings(texts: List[str], model_name: str) -> Tuple[SentenceTransformer, np.ndarray, NearestNeighbors]:
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    nn = NearestNeighbors(n_neighbors=min(10, len(texts)), metric="cosine")
    nn.fit(emb)
    return model, emb, nn


def rank_candidates_for_target(
    model: SentenceTransformer,
    emb: np.ndarray,
    nn: NearestNeighbors,
    corpus_texts: List[str],
    corpus_df: pd.DataFrame,
    target_text: str,
    top_k: int,
    exclude_url: Optional[str] = None,
) -> pd.DataFrame:
    q_emb = model.encode([target_text], normalize_embeddings=True)
    distances, indices = nn.kneighbors(q_emb, n_neighbors=min(top_k + 5, len(corpus_texts)))
    rows = []
    for d, i in zip(distances[0], indices[0]):
        url = corpus_df.iloc[i]["url"]
        if exclude_url and url == exclude_url:
            continue
        rows.append({
            "candidate_url": url,
            "candidate_title": corpus_df.iloc[i]["title"],
            "score": float(1.0 - d),  # cosine similarity ~ 1 - distance
        })
        if len(rows) >= top_k:
            break
    out = pd.DataFrame(rows)
    if not out.empty:
        out["suggested_anchor"] = out["candidate_title"].str.strip()
    return out


def suggest_for_all(
    pages_csv: str,
    output_csv: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
):
    df = load_pages(pages_csv)
    # Represent each page by title + first 512 chars of content for practical relevance
    texts = (df["title"].fillna("") + ". " + df["content"].fillna("").str.slice(0, 512)).tolist()
    model, emb, nn = build_embeddings(texts, model_name)

    all_rows = []
    for idx, row in df.iterrows():
        target_text = texts[idx]
        candidates = rank_candidates_for_target(
            model, emb, nn, texts, df, target_text, top_k=top_k, exclude_url=row["url"]
        )
        candidates.insert(0, "source_url", row["url"])
        candidates.insert(1, "source_title", row["title"])
        all_rows.append(candidates)

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


def suggest_for_target(
    pages_csv: str,
    output_csv: str,
    target_url: Optional[str] = None,
    target_topic: Optional[str] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
):
    df = load_pages(pages_csv)
    texts = (df["title"].fillna("") + ". " + df["content"].fillna("").str.slice(0, 512)).tolist()
    model, emb, nn = build_embeddings(texts, model_name)

    if target_url:
        if target_url not in set(df["url"]):
            raise ValueError(f"target_url not found in CSV: {target_url}")
        row = df[df["url"] == target_url].iloc[0]
        idx = row.name
        target_text = texts[idx]
        source_url = row["url"]
        source_title = row["title"]
        exclude_url = source_url
    elif target_topic:
        target_text = target_topic
        source_url = ""
        source_title = target_topic
        exclude_url = None
    else:
        raise ValueError("Provide either --target_url or --target_topic")

    candidates = rank_candidates_for_target(
        model, emb, nn, texts, df, target_text, top_k=top_k, exclude_url=exclude_url
    )
    candidates.insert(0, "source_url", source_url)
    candidates.insert(1, "source_title", source_title)
    candidates.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_all = sub.add_parser("batch", help="Suggest for all pages in the CSV")
    p_all.add_argument("--pages", default="data/pages.csv")
    p_all.add_argument("--output", default="data/internal_links_batch.csv")
    p_all.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_all.add_argument("--top_k", type=int, default=5)

    p_one = sub.add_parser("target", help="Suggest for a single target page/topic")
    p_one.add_argument("--pages", default="data/pages.csv")
    p_one.add_argument("--output", default="data/internal_links_target.csv")
    p_one.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_one.add_argument("--top_k", type=int, default=5)
    p_one.add_argument("--target_url", default=None)
    p_one.add_argument("--target_topic", default=None)

    args = parser.parse_args()
    if args.mode == "batch":
        suggest_for_all(args.pages, args.output, args.model, args.top_k)
    else:
        suggest_for_target(args.pages, args.output, args.target_url, args.target_topic, args.model, args.top_k)


