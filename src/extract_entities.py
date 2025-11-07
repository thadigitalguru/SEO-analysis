"""
Extract named entities (ORG, LOC, PER, MISC) from SERP snippets to enrich briefs.
"""
import argparse
import pandas as pd
from transformers import pipeline
from src.utils import read_serp_csv, save_csv

def extract_entities(input_csv: str, output_csv: str, model_name: str = "dslim/bert-base-NER"):
    nlp = pipeline("ner", model=model_name, aggregation_strategy="simple")
    df = read_serp_csv(input_csv)
    ents = []
    for _, row in df.iterrows():
        text = f"{row['title']}. {row['snippet']}"
        items = nlp(text)
        ents.append({
            "query": row["query"],
            "entities": "; ".join([f\"{e['word']}({e['entity_group']})\" for e in items])
        })
    out = pd.DataFrame(ents)
    save_csv(out, output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/examples/serp_samples.csv")
    parser.add_argument("--output", default="data/examples/serp_entities.csv")
    parser.add_argument("--model", default="dslim/bert-base-NER")
    args = parser.parse_args()
    extract_entities(args.input, args.output, args.model)