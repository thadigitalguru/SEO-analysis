import json
from pathlib import Path
import pandas as pd
import pytest
from src.schema_org import (
    generate_article,
    generate_product,
    generate_faqpage,
    validate_jsonld,
    generate_from_csv,
)


def test_generate_article():
    row = pd.Series({
        "url": "https://example.com/article",
        "headline": "Test Article",
        "author": "John Doe",
        "datePublished": "2024-01-01",
        "description": "A test article",
    })
    schema = generate_article(row)
    assert schema["@context"] == "https://schema.org"
    assert schema["@type"] == "Article"
    assert schema["headline"] == "Test Article"
    assert schema["author"]["name"] == "John Doe"
    assert schema["datePublished"] == "2024-01-01"


def test_generate_product():
    row = pd.Series({
        "url": "https://example.com/product",
        "name": "Test Product",
        "price": "29.99",
        "priceCurrency": "USD",
        "brand": "TestBrand",
    })
    schema = generate_product(row)
    assert schema["@type"] == "Product"
    assert schema["name"] == "Test Product"
    assert schema["offers"]["price"] == "29.99"
    assert schema["offers"]["priceCurrency"] == "USD"
    assert schema["brand"]["name"] == "TestBrand"


def test_generate_faqpage():
    df = pd.DataFrame({
        "url": ["https://example.com/faq"] * 2,
        "question": ["Q1", "Q2"],
        "answer": ["A1", "A2"],
    })
    schema = generate_faqpage(df)
    assert schema["@type"] == "FAQPage"
    assert len(schema["mainEntity"]) == 2
    assert schema["mainEntity"][0]["@type"] == "Question"
    assert schema["mainEntity"][0]["name"] == "Q1"
    assert schema["mainEntity"][0]["acceptedAnswer"]["text"] == "A1"


def test_validate_article_ok(tmp_path: Path):
    schema = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": "Test",
        "author": {"@type": "Person", "name": "Author"},
    }
    file_path = tmp_path / "test.jsonld"
    with open(file_path, "w") as f:
        json.dump(schema, f)
    path, ok, errors = validate_jsonld(str(file_path), "Article")
    assert ok
    assert len(errors) == 0


def test_validate_article_missing_fields(tmp_path: Path):
    schema = {
        "@context": "https://schema.org",
        "@type": "Article",
    }
    file_path = tmp_path / "test.jsonld"
    with open(file_path, "w") as f:
        json.dump(schema, f)
    path, ok, errors = validate_jsonld(str(file_path), "Article")
    assert not ok
    assert any("headline" in e.lower() for e in errors)
    assert any("author" in e.lower() for e in errors)


def test_validate_faqpage_ok(tmp_path: Path):
    schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": "Q1",
                "acceptedAnswer": {"@type": "Answer", "text": "A1"},
            }
        ],
    }
    file_path = tmp_path / "test.jsonld"
    with open(file_path, "w") as f:
        json.dump(schema, f)
    path, ok, errors = validate_jsonld(str(file_path), "FAQPage")
    assert ok


def test_validate_faqpage_invalid(tmp_path: Path):
    schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [{"@type": "Question"}],  # missing name and answer
    }
    file_path = tmp_path / "test.jsonld"
    with open(file_path, "w") as f:
        json.dump(schema, f)
    path, ok, errors = validate_jsonld(str(file_path), "FAQPage")
    assert not ok
    assert any("name" in e.lower() for e in errors)


def test_generate_from_csv_article(tmp_path: Path):
    csv_path = tmp_path / "articles.csv"
    pd.DataFrame({
        "url": ["https://example.com/a"],
        "headline": ["Test"],
        "author": ["Author"],
    }).to_csv(csv_path, index=False)
    output_dir = tmp_path / "output"
    files = generate_from_csv(str(csv_path), "Article", str(output_dir))
    assert len(files) == 1
    assert files[0].exists()
    with open(files[0]) as f:
        data = json.load(f)
        assert data["@type"] == "Article"


def test_generate_from_csv_faqpage_grouped(tmp_path: Path):
    csv_path = tmp_path / "faqs.csv"
    pd.DataFrame({
        "url": ["https://example.com/faq", "https://example.com/faq"],
        "question": ["Q1", "Q2"],
        "answer": ["A1", "A2"],
    }).to_csv(csv_path, index=False)
    output_dir = tmp_path / "output"
    files = generate_from_csv(str(csv_path), "FAQPage", str(output_dir))
    assert len(files) == 1
    with open(files[0]) as f:
        data = json.load(f)
        assert data["@type"] == "FAQPage"
        assert len(data["mainEntity"]) == 2

