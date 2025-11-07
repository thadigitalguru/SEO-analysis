"""
Sitemap utilities: generate XML sitemaps, validate and compare.

Features
- generate: Build sitemap files split by N URLs, optional gzip, and a sitemap index.
- validate: Basic structure validation for sitemap or sitemap index files.
- compare: Compare sitemap URLs vs a canonical URL list (CSV), report orphans/rogues.

Inputs
- For generation and compare: CSV with columns: url[, lastmod]

Outputs
- Generated sitemaps under an output directory (default: public/sitemaps/)
- Reports printed to stdout and optional CSV outputs for orphans/rogues
"""
import argparse
import datetime as dt
import gzip
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import pandas as pd
from urllib.request import urlopen
from urllib.error import URLError, HTTPError


NSMAP = {
    "sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9",
}
ET.register_namespace("", NSMAP["sitemap"])  # default namespace


def read_pages_csv(csv_path: str, keep_datetime: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "url" not in df.columns:
        raise ValueError("CSV must include a 'url' column")
    # normalize optional lastmod to W3C Datetime (YYYY-MM-DD)
    if "lastmod" in df.columns:
        def _norm(x):
            try:
                ts = pd.to_datetime(x, errors="coerce") if pd.notna(x) else None
                if ts is None or pd.isna(ts):
                    return None
                if keep_datetime:
                    # use Zulu (no tz if ambiguous)
                    return ts.strftime("%Y-%m-%dT%H:%M:%S")
                return ts.date().isoformat()
            except Exception:
                return None
        df["lastmod"] = df["lastmod"].apply(_norm)
    return df


def chunked(iterable: Iterable, size: int) -> Iterable[List]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _build_urlset(url_rows: List[Tuple[str, Optional[str], Optional[str], Optional[float]]]) -> ET.Element:
    urlset = ET.Element(ET.QName(NSMAP["sitemap"], "urlset"))
    for loc, lastmod, changefreq, priority in url_rows:
        url_el = ET.SubElement(urlset, ET.QName(NSMAP["sitemap"], "url"))
        loc_el = ET.SubElement(url_el, ET.QName(NSMAP["sitemap"], "loc"))
        loc_el.text = loc
        if lastmod:
            lm_el = ET.SubElement(url_el, ET.QName(NSMAP["sitemap"], "lastmod"))
            lm_el.text = lastmod
        if changefreq:
            cf_el = ET.SubElement(url_el, ET.QName(NSMAP["sitemap"], "changefreq"))
            cf_el.text = str(changefreq)
        if priority is not None:
            try:
                pr = float(priority)
                if 0.0 <= pr <= 1.0:
                    pr_el = ET.SubElement(url_el, ET.QName(NSMAP["sitemap"], "priority"))
                    pr_el.text = f"{pr:.1f}"
            except Exception:
                pass
    return urlset


def _write_xml(tree: ET.ElementTree, path: Path, gzip_enabled: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    xml_bytes = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    if gzip_enabled:
        gz_path = path.with_suffix(path.suffix + ".gz")
        with gzip.open(gz_path, "wb") as f:
            f.write(xml_bytes)
        return gz_path
    else:
        with open(path, "wb") as f:
            f.write(xml_bytes)
        return path


def generate_sitemaps(
    pages_csv: str,
    out_dir: str = "public/sitemaps",
    max_urls_per_file: int = 50000,
    base_index_url: Optional[str] = None,
    gzip_enabled: bool = False,
    lastmod_datetime: bool = False,
) -> List[Path]:
    df = read_pages_csv(pages_csv, keep_datetime=lastmod_datetime)
    # optional changefreq/priority passthrough if present
    changefreqs = df["changefreq"].tolist() if "changefreq" in df.columns else [None] * len(df)
    priorities = df["priority"].tolist() if "priority" in df.columns else [None] * len(df)
    lastmods = df["lastmod"].tolist() if "lastmod" in df.columns else [None] * len(df)
    rows = list(zip(df["url"].tolist(), lastmods, changefreqs, priorities))

    out_dir_p = Path(out_dir)
    created_files: List[Path] = []

    # generate urlset files
    for i, batch in enumerate(chunked(rows, max_urls_per_file), start=1):
        urlset = _build_urlset(batch)
        tree = ET.ElementTree(urlset)
        file_path = out_dir_p / f"sitemap-{i}.xml"
        written = _write_xml(tree, file_path, gzip_enabled)
        created_files.append(written)

    # generate index
    index = ET.Element(ET.QName(NSMAP["sitemap"], "sitemapindex"))
    now = dt.datetime.utcnow().date().isoformat()
    for f in created_files:
        sm_el = ET.SubElement(index, ET.QName(NSMAP["sitemap"], "sitemap"))
        loc_el = ET.SubElement(sm_el, ET.QName(NSMAP["sitemap"], "loc"))
        if base_index_url:
            # determine file name with gzip if present
            loc_el.text = urljoin(base_index_url.rstrip("/") + "/", f.name)
        else:
            loc_el.text = f.as_posix()
        lm_el = ET.SubElement(sm_el, ET.QName(NSMAP["sitemap"], "lastmod"))
        lm_el.text = now

    index_tree = ET.ElementTree(index)
    index_path = out_dir_p / "sitemap.xml"
    written_index = _write_xml(index_tree, index_path, gzip_enabled=False)  # index typically not gzipped
    created_files.insert(0, written_index)

    print(f"Generated {len(created_files)-1} sitemaps + index at: {out_dir_p}")
    return created_files


def validate_sitemap_file(path: str) -> Tuple[str, bool, str]:
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        tag = root.tag
        if tag.endswith("urlset"):
            urls = root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url")
            if not urls:
                return path, False, "urlset has no <url> entries"
            for u in urls:
                loc = u.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                if loc is None or not (loc.text and loc.text.startswith("http")):
                    return path, False, "missing/invalid <loc> in url"
            return path, True, f"ok: {len(urls)} urls"
        elif tag.endswith("sitemapindex"):
            items = root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap")
            if not items:
                return path, False, "sitemapindex has no <sitemap> entries"
            for s in items:
                loc = s.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
                if loc is None or not (loc.text and loc.text.startswith("http")):
                    return path, False, "missing/invalid <loc> in sitemap"
            return path, True, f"ok: {len(items)} child sitemaps"
        else:
            return path, False, f"unknown root tag: {tag}"
    except ET.ParseError as e:
        return path, False, f"parse error: {e}"
    except FileNotFoundError:
        return path, False, "file not found"


def _parse_xml_bytes(data: bytes) -> ET.Element:
    return ET.fromstring(data)


def _fetch_xml(url: str, timeout: int = 10) -> Optional[ET.Element]:
    try:
        with urlopen(url, timeout=timeout) as resp:
            return _parse_xml_bytes(resp.read())
    except (URLError, HTTPError, TimeoutError, ValueError):
        return None


def _collect_sitemap_urls_from_root(root: ET.Element) -> List[str]:
    urls: List[str] = []
    if root.tag.endswith("urlset"):
        for u in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}url"):
            loc = u.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
            if loc is not None and loc.text:
                urls.append(loc.text.strip())
    elif root.tag.endswith("sitemapindex"):
        for s in root.findall("{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"):
            loc = s.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")
            if loc is not None and loc.text:
                urls.append(loc.text.strip())
    return urls


def compare_sitemap_vs_urls(
    sitemaps: List[str],
    urls_csv: str,
    expand_index: bool = False,
    timeout: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # gather all <loc> URLs from provided sitemap files
    sm_urls: List[str] = []
    to_expand: List[str] = []
    for p in sitemaps:
        try:
            if p.startswith("http://") or p.startswith("https://"):
                root = _fetch_xml(p, timeout=timeout)
                if root is None:
                    continue
            else:
                if p.endswith(".gz"):
                    with gzip.open(p, "rb") as f:
                        root = _parse_xml_bytes(f.read())
                else:
                    root = ET.parse(p).getroot()
        except Exception:
            continue
        urls = _collect_sitemap_urls_from_root(root)
        if root.tag.endswith("urlset"):
            sm_urls.extend(urls)
        else:
            # sitemapindex
            if expand_index:
                to_expand.extend(urls)
            else:
                sm_urls.extend(urls)

    # Expand index children if requested (best-effort)
    if expand_index and to_expand:
        for url in to_expand:
            root = _fetch_xml(url, timeout=timeout) if (url.startswith("http://") or url.startswith("https://")) else None
            if root is None:
                # try local path
                try:
                    if url.endswith(".gz"):
                        with gzip.open(url, "rb") as f:
                            root = _parse_xml_bytes(f.read())
                    else:
                        root = ET.parse(url).getroot()
                except Exception:
                    continue
            sm_urls.extend(_collect_sitemap_urls_from_root(root))

    df_urls = read_pages_csv(urls_csv)
    site_urls = set(df_urls["url"].astype(str).str.strip().tolist())
    sm_set = set(sm_urls)

    # orphans: in site list but not in sitemap
    orphan = sorted(site_urls - sm_set)
    # rogues: in sitemap but not in site list
    rogue = sorted(sm_set - site_urls)

    orphan_df = pd.DataFrame({"url": orphan})
    rogue_df = pd.DataFrame({"url": rogue})
    return orphan_df, rogue_df


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate sitemap files + index")
    p_gen.add_argument("--pages", required=True, help="CSV with 'url[,lastmod]'")
    p_gen.add_argument("--out_dir", default="public/sitemaps")
    p_gen.add_argument("--max_urls", type=int, default=50000)
    p_gen.add_argument("--base_index_url", default=None, help="Base URL to prefix in sitemap index <loc>")
    p_gen.add_argument("--gzip", action="store_true")
    p_gen.add_argument("--lastmod_datetime", action="store_true")

    p_val = sub.add_parser("validate", help="Validate a sitemap or sitemap index file")
    p_val.add_argument("--path", required=True)

    p_cmp = sub.add_parser("compare", help="Compare sitemap URLs to a CSV of site URLs")
    p_cmp.add_argument("--sitemaps", nargs="+", help="Paths to sitemap files (can include .gz)")
    p_cmp.add_argument("--pages", required=True, help="CSV with 'url[,lastmod]'")
    p_cmp.add_argument("--orphans_out", default=None)
    p_cmp.add_argument("--rogues_out", default=None)
    p_cmp.add_argument("--expand_index", action="store_true", help="If a sitemap index is provided, fetch child sitemaps and expand page URLs")
    p_cmp.add_argument("--timeout", type=int, default=10)

    args = parser.parse_args()
    if args.cmd == "generate":
        created = generate_sitemaps(args.pages, args.out_dir, args.max_urls, args.base_index_url, args.gzip, args.lastmod_datetime)
        for p in created:
            print(p)
    elif args.cmd == "validate":
        path, ok, msg = validate_sitemap_file(args.path)
        print(f"{path}\t{ok}\t{msg}")
    elif args.cmd == "compare":
        orphan_df, rogue_df = compare_sitemap_vs_urls(args.sitemaps, args.pages, args.expand_index, args.timeout)
        print(f"Orphans: {len(orphan_df)}\tRogues: {len(rogue_df)}")
        if args.orphans_out:
            Path(args.orphans_out).parent.mkdir(parents=True, exist_ok=True)
            orphan_df.to_csv(args.orphans_out, index=False)
            print(f"Saved orphans: {args.orphans_out}")
        if args.rogues_out:
            Path(args.rogues_out).parent.mkdir(parents=True, exist_ok=True)
            rogue_df.to_csv(args.rogues_out, index=False)
            print(f"Saved rogues: {args.rogues_out}")


if __name__ == "__main__":
    main()


