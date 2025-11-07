"""
Generate SEO titles and meta descriptions for each cluster using T5-small.
"""
import argparse
import pandas as pd
from transformers import pipeline
from src.utils import save_csv

PROMPT_TMPL = """Generate an SEO page title (max ~60 chars) and a meta description (150-160 chars)
for the topic: "{topic}".
Tone: authoritative but friendly. Brand: neutral.
Return as: Title: <title> || Meta: <meta>
"""

def parse_output(text: str):
    parts = text.split("||")
    title = ""
    meta = ""
    if len(parts) >= 2:
        title = parts[0].replace("Title:", "").strip()
        meta = parts[1].replace("Meta:", "").strip()
    else:
        s = text.strip().split(". ")
        title = s[0][:60]
        meta = text[:160]
    return title, meta

def generate_meta(input_csv: str, output_csv: str, model_name: str = "google/flan-t5-small", max_new_tokens: int = 96):
    df = pd.read_csv(input_csv)
    if "cluster_rep" not in df.columns:
        raise ValueError("Input should include a 'cluster_rep' column (run clustering first).")
    gen = pipeline("text2text-generation", model=model_name)
    rows = []
    for topic in sorted(df["cluster_rep"].unique()):
        prompt = PROMPT_TMPL.format(topic=topic)
        res = gen(prompt, max_new_tokens=max_new_tokens)[0]["generated_text"]
        title, meta = parse_output(res)
        rows.append({"cluster_rep": topic, "title": title, "meta_description": meta})
    out = pd.DataFrame(rows)
    save_csv(out, output_csv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/examples/keywords_clusters.csv")
    parser.add_argument("--output", default="data/examples/seo_titles_meta.csv")
    parser.add_argument("--model", default="google/flan-t5-small")
    args = parser.parse_args()
    generate_meta(args.input, args.output, args.model)