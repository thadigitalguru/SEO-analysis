"""
Classify search intent for each keyword using zero-shot classification.
Labels: informational, navigational, commercial, transactional
"""
import argparse
import pandas as pd
from transformers import pipeline
from src.utils import read_keywords_csv, save_csv

LABELS = ["informational", "navigational", "commercial", "transactional"]

def classify_intent(input_csv: str, output_csv: str, model_name: str = "facebook/bart-large-mnli"):
    df = read_keywords_csv(input_csv)
    clf = pipeline("zero-shot-classification", model=model_name)
    intents = []
    for kw in df["keyword"].tolist():
        res = clf(kw, LABELS, multi_label=False)
        label = res["labels"][0]
        score = float(res["scores"][0])
        intents.append({"keyword": kw, "intent": label, "confidence": round(score, 4)})
    out = pd.DataFrame(intents)
    merged = df.merge(out, on="keyword", how="left")
    save_csv(merged, output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/examples/keywords.csv")
    parser.add_argument("--output", default="data/examples/keywords_intent.csv")
    parser.add_argument("--model", default="facebook/bart-large-mnli")
    args = parser.parse_args()
    classify_intent(args.input, args.output, args.model)