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

### Internal linking suggestions (embeddings-assisted)
CSV input must include: `url,title,content`

```bash
# Suggest for every page (exclude self); writes top-N candidates per page
python -m src.internal_linking batch --pages data/pages.csv --output data/internal_links_batch.csv --top_k 5 --content_chars 512 --min_content_len 50 --anchor_from_sentence

# Suggest for a single target page (by URL)
python -m src.internal_linking target --pages data/pages.csv --output data/internal_links_target.csv --target_url https://example.com/running-shoes --anchor_from_sentence

# Or by topic text if the page is not in the CSV
python -m src.internal_linking target --pages data/pages.csv --output data/internal_links_target.csv --target_topic "marathon training nutrition"
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
- `data/internal_links_*.csv` — internal link candidates with anchor suggestions.

### Sitemap utilities
CSV input for generation/compare must include: `url[,lastmod]`

```bash
# Generate sitemaps split by N URLs + sitemap index
python -m src.sitemap_utils generate --pages data/pages.csv --out_dir public/sitemaps --max_urls 50000 --base_index_url https://www.example.com/sitemaps/ --lastmod_datetime

# Validate a sitemap (or index) file
python -m src.sitemap_utils validate --path public/sitemaps/sitemap.xml

# Compare sitemap URLs vs canonical URL list (orphans/rogues)
python -m src.sitemap_utils compare --sitemaps public/sitemaps/sitemap.xml --pages data/pages.csv --expand_index --timeout 10 --orphans_out reports/orphans.csv --rogues_out reports/rogues.csv
```

### Core Web Vitals snapshot (PageSpeed Insights)
Requires a Google PSI API key in env `GOOGLE_PSI_KEY` or pass `--api_key`.

Input CSV must include: `url`

```bash
# Fetch CWV for a sample of URLs (mobile + desktop), concurrency 5
export GOOGLE_PSI_KEY=your_key_here
python -m src.psi_snapshot --urls_csv data/pages.csv --output_csv reports/psi_snapshot.csv --strategies mobile desktop --sample_n 50 --concurrency 5
```

Outputs columns: `url, strategy, overall_category, LCP/INP/CLS/FID/FCP/TTFB (values + categories)`. Field data (CrUX) is preferred; Lighthouse audits are used as fallback when field metrics are unavailable.

### Schema.org JSON-LD generator/validator
Generate and validate structured data for Article, Product, and FAQPage.

CSV input columns vary by type:
- **Article**: `url, headline, author` (required); `datePublished, dateModified, image, description, publisher` (optional)
- **Product**: `url, name` (required); `description, image, price, priceCurrency, brand, sku` (optional)
- **FAQPage**: `url, question, answer` (required; multiple rows per page grouped by URL)

```bash
# Generate JSON-LD from CSV
python -m src.schema_org generate --csv data/articles.csv --type Article --output_dir public/schema
python -m src.schema_org generate --csv data/products.csv --type Product --output_dir public/schema
python -m src.schema_org generate --csv data/faqs.csv --type FAQPage --output_dir public/schema

# Validate JSON-LD file(s)
python -m src.schema_org validate --path public/schema/article_1.jsonld --type Article
python -m src.schema_org validate --path public/schema/  # validates all .jsonld files in directory
```

Via CLI wrapper:
```bash
seo-tools schema generate --csv data/articles.csv --type Article --output_dir public/schema
seo-tools schema validate --path public/schema/article_1.jsonld
```

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