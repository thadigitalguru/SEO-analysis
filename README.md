# Transformers for Marketers & Technical SEO — VS Code Beginner Project

Learn modern NLP in a practical, marketer-focused way using Hugging Face 🤗.
You’ll build small, real-world tools: intent tagging, keyword clustering, title & meta generation,
entity extraction, summarization, and a tiny semantic search — all in Python.

## Quickstart
1) Open this folder in VS Code.
2) Create a venv & install deps: Run task **Create venv + install** (from `Tasks: Run Task`) or:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3) Copy `.env.example` to `.env` and (optionally) add your Hugging Face token.
4) Explore **notebooks/** to learn, then run end‑to‑end **src/** scripts.

### Run the scripts (examples)
Activate the venv first:
```bash
source .venv/bin/activate
```

Then run:
```bash
# 1) Intent classification
python -m src.intent_classifier --input data/examples/keywords.csv --output data/examples/keywords_intent.csv

# 2) Keyword clustering
python -m src.cluster_keywords --input data/examples/keywords.csv --output data/examples/keywords_clusters.csv --clusters 5

# 3) SEO titles + meta from clusters
python -m src.generate_meta --input data/examples/keywords_clusters.csv --output data/examples/seo_titles_meta.csv

# 4) NER from SERP samples
python -m src.extract_entities --input data/examples/serp_samples.csv --output data/examples/serp_entities.csv

# 5) Tiny semantic search demo
python -m src.semantic_search --csv data/examples/serp_samples.csv --query "marathon shoe tips for beginners"
```

### Model picks (CPU‑friendly-ish)
- Zero‑shot intent: `facebook/bart-large-mnli` (accurate; slower) or a smaller MNLI model.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (fast & high‑quality for size).
- Summarization: `sshleifer/distilbart-cnn-12-6` (compact).
- NER: `dslim/bert-base-NER`.
- Generation (titles/meta): `google/flan-t5-small`.

### Learning path
1. Intent → 2. Clustering → 3. Titles/Meta → 4. NER → 5. Summarization → 6. Semantic search.

## Outputs (example run)
- `data/examples/keywords_intent.csv` — zero‑shot intents.
- `data/examples/keywords_clusters.csv` — cluster ids + representatives.
- `data/examples/seo_titles_meta.csv` — generated titles and meta descriptions.

## Version control
This repo includes a `.gitignore` to keep virtual envs and caches out of git. To save your work:
```bash
git init -b main
git add .
git commit -m "Initial commit: setup + deps + sample outputs"
```
To publish to GitHub, create a repo and push:
```bash
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```