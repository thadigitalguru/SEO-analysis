"""
Schema.org JSON-LD generator and validator for Article, Product, FAQPage.

- generate: Build JSON-LD from CSV input (one row per entity)
- validate: Check JSON-LD files for required fields and structure

Input CSV columns vary by type:
- Article: url, headline, author, datePublished, dateModified, image
- Product: url, name, description, image, price, priceCurrency, brand
- FAQPage: url, question, answer (multiple rows per page)
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from src.utils import atomic_write, utc_now_iso


REQUIRED_FIELDS = {
    "Article": ["url", "headline", "author"],
    "Product": ["url", "name"],
    "FAQPage": ["url", "question", "answer"],
}


def generate_article(row: pd.Series) -> Dict[str, Any]:
    schema = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": str(row["headline"]),
        "author": {
            "@type": "Person",
            "name": str(row["author"]),
        },
    }
    if "url" in row and pd.notna(row["url"]):
        schema["mainEntityOfPage"] = {"@type": "WebPage", "@id": str(row["url"])}
    if "datePublished" in row and pd.notna(row["datePublished"]):
        schema["datePublished"] = str(row["datePublished"])
    if "dateModified" in row and pd.notna(row["dateModified"]):
        schema["dateModified"] = str(row["dateModified"])
    if "image" in row and pd.notna(row["image"]):
        schema["image"] = str(row["image"])
    if "description" in row and pd.notna(row["description"]):
        schema["description"] = str(row["description"])
    if "publisher" in row and pd.notna(row["publisher"]):
        schema["publisher"] = {
            "@type": "Organization",
            "name": str(row["publisher"]),
        }
    return schema


def generate_product(row: pd.Series) -> Dict[str, Any]:
    schema = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": str(row["name"]),
    }
    if "url" in row and pd.notna(row["url"]):
        schema["url"] = str(row["url"])
    if "description" in row and pd.notna(row["description"]):
        schema["description"] = str(row["description"])
    if "image" in row and pd.notna(row["image"]):
        schema["image"] = str(row["image"])
    if "price" in row and pd.notna(row["price"]):
        schema["offers"] = {
            "@type": "Offer",
            "price": str(row["price"]),
        }
        if "priceCurrency" in row and pd.notna(row["priceCurrency"]):
            schema["offers"]["priceCurrency"] = str(row["priceCurrency"])
    if "brand" in row and pd.notna(row["brand"]):
        schema["brand"] = {
            "@type": "Brand",
            "name": str(row["brand"]),
        }
    if "sku" in row and pd.notna(row["sku"]):
        schema["sku"] = str(row["sku"])
    return schema


def generate_faqpage(df: pd.DataFrame) -> Dict[str, Any]:
    # Group by URL if present, otherwise single FAQPage
    if "url" in df.columns:
        url = df["url"].iloc[0] if len(df) > 0 else None
    else:
        url = None

    faqs = []
    for _, row in df.iterrows():
        faq = {
            "@type": "Question",
            "name": str(row["question"]),
            "acceptedAnswer": {
                "@type": "Answer",
                "text": str(row["answer"]),
            },
        }
        faqs.append(faq)

    schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": faqs,
    }
    if url and pd.notna(url):
        schema["mainEntityOfPage"] = {"@type": "WebPage", "@id": str(url)}
    return schema


def generate_from_csv(csv_path: str, schema_type: str, output_dir: str) -> List[Path]:
    df = pd.read_csv(csv_path)
    required = REQUIRED_FIELDS.get(schema_type)
    if not required:
        raise ValueError(f"Unknown schema type: {schema_type}. Supported: {list(REQUIRED_FIELDS.keys())}")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {schema_type}: {missing}")

    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)
    created_files: List[Path] = []

    if schema_type == "FAQPage":
        # Group by URL if present
        if "url" in df.columns:
            for url, group in df.groupby("url"):
                schema = generate_faqpage(group)
                filename = f"faqpage_{url.replace('https://', '').replace('http://', '').replace('/', '_')}.jsonld"
                file_path = output_dir_p / filename
                data = json.dumps(schema, indent=2, ensure_ascii=False).encode("utf-8")
                atomic_write(data, str(file_path))
                created_files.append(file_path)
        else:
            schema = generate_faqpage(df)
            file_path = output_dir_p / "faqpage.jsonld"
            data = json.dumps(schema, indent=2, ensure_ascii=False).encode("utf-8")
            atomic_write(data, str(file_path))
            created_files.append(file_path)
    else:
        # One JSON-LD per row
        for idx, row in df.iterrows():
            if schema_type == "Article":
                schema = generate_article(row)
            elif schema_type == "Product":
                schema = generate_product(row)
            else:
                continue

            # Generate filename from URL or index
            if "url" in row and pd.notna(row["url"]):
                url_str = str(row["url"]).replace("https://", "").replace("http://", "").replace("/", "_")
                filename = f"{schema_type.lower()}_{url_str}.jsonld"
            else:
                filename = f"{schema_type.lower()}_{idx}.jsonld"
            file_path = output_dir_p / filename
            data = json.dumps(schema, indent=2, ensure_ascii=False).encode("utf-8")
            atomic_write(data, str(file_path))
            created_files.append(file_path)

    print(f"Generated {len(created_files)} JSON-LD files for {schema_type} at: {output_dir_p}")
    return created_files


def validate_jsonld(file_path: str, schema_type: Optional[str] = None) -> Tuple[str, bool, List[str]]:
    errors: List[str] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return file_path, False, [f"Invalid JSON: {e}"]
    except FileNotFoundError:
        return file_path, False, ["File not found"]

    # Check @context
    if "@context" not in data:
        errors.append("Missing @context")
    elif data["@context"] != "https://schema.org":
        errors.append(f"Invalid @context: {data['@context']}")

    # Check @type
    if "@type" not in data:
        errors.append("Missing @type")
        return file_path, False, errors
    detected_type = data["@type"]
    if schema_type and detected_type != schema_type:
        errors.append(f"Type mismatch: expected {schema_type}, got {detected_type}")

    # Validate required fields per type
    required = REQUIRED_FIELDS.get(detected_type, [])
    if detected_type == "Article":
        if "headline" not in data:
            errors.append("Missing required field: headline")
        if "author" not in data:
            errors.append("Missing required field: author")
    elif detected_type == "Product":
        if "name" not in data:
            errors.append("Missing required field: name")
    elif detected_type == "FAQPage":
        if "mainEntity" not in data:
            errors.append("Missing required field: mainEntity")
        elif not isinstance(data["mainEntity"], list):
            errors.append("mainEntity must be a list")
        else:
            for i, item in enumerate(data["mainEntity"]):
                if "@type" not in item or item["@type"] != "Question":
                    errors.append(f"mainEntity[{i}] must be a Question")
                if "name" not in item:
                    errors.append(f"mainEntity[{i}] missing 'name' (question)")
                if "acceptedAnswer" not in item:
                    errors.append(f"mainEntity[{i}] missing 'acceptedAnswer'")
                elif "@type" not in item.get("acceptedAnswer", {}) or item["acceptedAnswer"]["@type"] != "Answer":
                    errors.append(f"mainEntity[{i}].acceptedAnswer must be an Answer")

    return file_path, len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate JSON-LD from CSV")
    p_gen.add_argument("--csv", required=True, help="Input CSV")
    p_gen.add_argument("--type", choices=["Article", "Product", "FAQPage"], required=True)
    p_gen.add_argument("--output_dir", default="public/schema")

    p_val = sub.add_parser("validate", help="Validate JSON-LD file(s)")
    p_val.add_argument("--path", required=True, help="JSON-LD file or directory")
    p_val.add_argument("--type", choices=["Article", "Product", "FAQPage"], default=None, help="Expected schema type (optional)")

    args = parser.parse_args()
    if args.cmd == "generate":
        generate_from_csv(args.csv, args.type, args.output_dir)
    elif args.cmd == "validate":
        path = Path(args.path)
        if path.is_file():
            file_path, ok, errors = validate_jsonld(str(path), args.type)
            if ok:
                print(f"{file_path}: OK")
            else:
                print(f"{file_path}: FAILED")
                for e in errors:
                    print(f"  - {e}")
        elif path.is_dir():
            jsonld_files = list(path.glob("*.jsonld"))
            all_ok = True
            for f in jsonld_files:
                file_path, ok, errors = validate_jsonld(str(f), args.type)
                if ok:
                    print(f"{file_path}: OK")
                else:
                    all_ok = False
                    print(f"{file_path}: FAILED")
                    for e in errors:
                        print(f"  - {e}")
            if not all_ok:
                exit(1)
        else:
            print(f"Path not found: {args.path}")
            exit(1)


if __name__ == "__main__":
    main()

