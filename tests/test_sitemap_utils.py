from pathlib import Path
import pandas as pd
from src.sitemap_utils import generate_sitemaps, validate_sitemap_file


def test_generate_and_validate(tmp_path: Path):
    pages = tmp_path / "pages.csv"
    pd.DataFrame({
        "url": [
            "https://example.com/",
            "https://example.com/a",
            "https://example.com/b",
        ],
        "lastmod": ["2024-01-01", "2024-01-02", "2024-01-03"],
    }).to_csv(pages, index=False)

    out_dir = tmp_path / "sitemaps"
    files = generate_sitemaps(str(pages), str(out_dir), max_urls_per_file=2, base_index_url="https://example.com/sitemaps/")
    assert any(f.name == "sitemap.xml" for f in files)

    # Validate index
    index = out_dir / "sitemap.xml"
    path, ok, msg = validate_sitemap_file(str(index))
    assert ok, msg
