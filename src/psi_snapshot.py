"""
Core Web Vitals snapshot via PageSpeed Insights (PSI) API.

- Inputs: CSV with column: url
- Output: CSV with per-URL (and per strategy) CWV metrics and categories.

Metrics captured (prefer CrUX field data, fallback to Lighthouse):
- LCP, INP, CLS, FID, FCP, TTFB + overall_category and per-metric categories when available

Auth:
- Provide API key via --api_key or env var GOOGLE_PSI_KEY
"""
import os
import argparse
import asyncio
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import httpx
from urllib.parse import urlencode
from src.utils import write_dataframe, utc_now_iso


PSI_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"


def get_api_key(cli_key: Optional[str]) -> str:
    key = cli_key or os.getenv("GOOGLE_PSI_KEY")
    if not key:
        raise ValueError("Provide PSI API key via --api_key or env GOOGLE_PSI_KEY")
    return key


def pick(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def parse_crux_metric(metrics: Dict[str, Any], key: str) -> Dict[str, Any]:
    m = metrics.get(key) or {}
    # v5 typically: { percentile: 1234, category: "GOOD|..." }
    return {
        "percentile": m.get("percentile"),
        "category": m.get("category"),
    }


def parse_lh_audit(audits: Dict[str, Any], audit_key: str) -> Optional[float]:
    a = audits.get(audit_key)
    if not a:
        return None
    val = a.get("numericValue")
    # values are typically in milliseconds (for times) or unitless for CLS
    return float(val) if val is not None else None


def _select_experience(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer page-level loadingExperience, fallback to originLoadingExperience
    le = pick(payload, "loadingExperience") or {}
    if not le.get("metrics"):
        le = pick(payload, "originLoadingExperience") or {}
    return le


def parse_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer CrUX field data (page), fallback to origin
    le = _select_experience(payload)
    le_metrics = le.get("metrics") or {}
    overall_category = le.get("overall_category")

    def get_field_ms(name: str) -> Optional[float]:
        v = parse_crux_metric(le_metrics, name).get("percentile")
        return float(v) if v is not None else None

    def get_field_cat(name: str) -> Optional[str]:
        return parse_crux_metric(le_metrics, name).get("category")

    # Field metrics (ms or unitless for CLS)
    lcp_ms = get_field_ms("LCP")
    inp_ms = get_field_ms("INP")
    cls = parse_crux_metric(le_metrics, "CLS").get("percentile")
    if cls is not None:
        cls = float(cls)
        # CLS from CrUX is unitless in [0, ~1]; some responses may return hundredths
        if cls > 1:
            cls = cls / 100.0
    fid_ms = get_field_ms("FID")
    fcp_ms = get_field_ms("FCP")
    ttfb_ms = get_field_ms("TTFB")

    lcp_cat = get_field_cat("LCP")
    inp_cat = get_field_cat("INP")
    cls_cat = get_field_cat("CLS")
    fid_cat = get_field_cat("FID")
    fcp_cat = get_field_cat("FCP")
    ttfb_cat = get_field_cat("TTFB")

    # Fallbacks from Lighthouse audits if field missing
    audits = pick(payload, "lighthouseResult.audits") or {}
    lcp_source = "field" if lcp_ms is not None else None
    inp_source = "field" if inp_ms is not None else None
    cls_source = "field" if cls is not None else None
    fid_source = "field" if fid_ms is not None else None
    fcp_source = "field" if fcp_ms is not None else None
    ttfb_source = "field" if ttfb_ms is not None else None

    if lcp_ms is None:
        lcp_ms = parse_lh_audit(audits, "largest-contentful-paint")
        if lcp_ms is not None:
            lcp_source = "lh"
    if inp_ms is None:
        # INP audit key in PSI v5: "experimental-interaction-to-next-paint" when available
        inp_ms = parse_lh_audit(audits, "experimental-interaction-to-next-paint")
        if inp_ms is not None:
            inp_source = "lh"
    if cls is None:
        cls = parse_lh_audit(audits, "cumulative-layout-shift")
        if cls is not None:
            cls_source = "lh"
    if fcp_ms is None:
        fcp_ms = parse_lh_audit(audits, "first-contentful-paint")
        if fcp_ms is not None:
            fcp_source = "lh"
    if ttfb_ms is None:
        ttfb_ms = parse_lh_audit(audits, "server-response-time")
        if ttfb_ms is not None:
            ttfb_source = "lh"
    if fid_ms is None:
        # No Lighthouse FID; leave None
        pass

    return {
        "lcp_ms": lcp_ms,
        "inp_ms": inp_ms,
        "cls": cls,
        "fid_ms": fid_ms,
        "fcp_ms": fcp_ms,
        "ttfb_ms": ttfb_ms,
        "lcp_source": lcp_source,
        "inp_source": inp_source,
        "cls_source": cls_source,
        "fid_source": fid_source,
        "fcp_source": fcp_source,
        "ttfb_source": ttfb_source,
        "overall_category": overall_category,
        "lcp_category": lcp_cat,
        "inp_category": inp_cat,
        "cls_category": cls_cat,
        "fid_category": fid_cat,
        "fcp_category": fcp_cat,
        "ttfb_category": ttfb_cat,
    }


async def fetch_psi(client: httpx.AsyncClient, url: str, api_key: str, strategy: str, category: str, locale: str) -> Dict[str, Any]:
    params = {
        "url": url,
        "key": api_key,
        "strategy": strategy,
        "category": category,
        "locale": locale,
    }
    # simple retries with backoff
    delay = 1.0
    for attempt in range(3):
        try:
            resp = await client.get(PSI_ENDPOINT, params=params, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            parsed = parse_response(data)
            parsed.update({"url": url, "strategy": strategy})
            return parsed
        except Exception as e:
            err = str(e)
            if attempt < 2:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            return {"url": url, "strategy": strategy, "error": err}


async def gather_psi(urls: List[str], api_key: str, strategies: List[str], concurrency: int, category: str, locale: str, rps: float) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    interval = 1.0 / rps if rps and rps > 0 else 0.0
    last_call = 0.0

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def worker(u: str, strat: str):
            async with sem:
                try:
                    nonlocal last_call
                    if interval > 0:
                        now = asyncio.get_event_loop().time()
                        wait = max(0.0, last_call + interval - now)
                        if wait > 0:
                            await asyncio.sleep(wait)
                        last_call = asyncio.get_event_loop().time()
                    res = await fetch_psi(client, u, api_key, strat, category, locale)
                    results.append(res)
                except Exception as e:
                    results.append({"url": u, "strategy": strat, "error": str(e)})

        tasks = [worker(u, s) for u in urls for s in strategies]
        await asyncio.gather(*tasks)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls_csv", required=True, help="CSV with column 'url'")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--strategies", nargs="+", default=["mobile", "desktop"])
    parser.add_argument("--category", default="performance")
    parser.add_argument("--locale", default="en_US")
    parser.add_argument("--sample_n", type=int, default=0, help="Optionally sample N URLs from input")
    parser.add_argument("--random_sample", action="store_true", help="Randomly sample N URLs (with --seed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--rps", type=float, default=0.0, help="Throttle requests per second (0=unlimited)")
    parser.add_argument("--output_format", choices=["csv","jsonl","parquet"], default="csv")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    key = get_api_key(args.api_key)
    df = pd.read_csv(args.urls_csv)
    if "url" not in df.columns:
        raise ValueError("Input must contain a 'url' column")
    urls = df["url"].astype(str).str.strip().tolist()
    if args.sample_n and args.sample_n > 0:
        if args.random_sample:
            urls = pd.Series(urls).sample(n=min(args.sample_n, len(urls)), random_state=args.seed, replace=False).tolist()
        else:
            urls = urls[: args.sample_n]

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    results = asyncio.run(gather_psi(urls, key, args.strategies, args.concurrency, args.category, args.locale, args.rps))
    out = pd.DataFrame(results)
    # order columns for readability
    preferred_cols = [
        "url", "strategy",
        "overall_category",
        "lcp_ms", "lcp_category",
        "inp_ms", "inp_category",
        "cls", "cls_category",
        "fid_ms", "fid_category",
        "fcp_ms", "fcp_category",
        "ttfb_ms", "ttfb_category",
        "lcp_source","inp_source","cls_source","fid_source","fcp_source","ttfb_source",
        "error",
    ]
    cols = [c for c in preferred_cols if c in out.columns] + [c for c in out.columns if c not in preferred_cols]
    out = out[cols]
    out["generated_at_utc"] = utc_now_iso()
    write_dataframe(out, args.output_csv, fmt=args.output_format, overwrite=args.overwrite, float_format='%.6f')
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()


