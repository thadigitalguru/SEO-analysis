import argparse
from src.intent_classifier import classify_intent
from src.cluster_keywords import cluster_keywords
from src.generate_meta import generate_meta
from src.extract_entities import extract_entities
from src.internal_linking import suggest_for_all, suggest_for_target
from src.sitemap_utils import generate_sitemaps, validate_sitemap_file, compare_sitemap_vs_urls
from src.psi_snapshot import main as psi_main


def main():
    parser = argparse.ArgumentParser(prog="seo-tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Intent
    p_intent = sub.add_parser("intent", help="Zero-shot intent classification")
    p_intent.add_argument("--input", required=True)
    p_intent.add_argument("--output", required=True)
    p_intent.add_argument("--model", default="facebook/bart-large-mnli")

    # Clustering
    p_cluster = sub.add_parser("cluster", help="Keyword clustering")
    p_cluster.add_argument("--input", required=True)
    p_cluster.add_argument("--output", required=True)
    p_cluster.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_cluster.add_argument("--clusters", type=int, default=5)

    # Meta generation
    p_meta = sub.add_parser("meta", help="Generate titles/meta from clusters")
    p_meta.add_argument("--input", required=True)
    p_meta.add_argument("--output", required=True)
    p_meta.add_argument("--model", default="google/flan-t5-small")

    # NER
    p_ner = sub.add_parser("ner", help="Extract entities from SERP CSV")
    p_ner.add_argument("--input", required=True)
    p_ner.add_argument("--output", required=True)
    p_ner.add_argument("--model", default="dslim/bert-base-NER")

    # Internal linking
    p_link = sub.add_parser("links", help="Internal linking suggestions")
    link_sub = p_link.add_subparsers(dest="mode", required=True)

    p_link_all = link_sub.add_parser("batch")
    p_link_all.add_argument("--pages", required=True)
    p_link_all.add_argument("--output", required=True)
    p_link_all.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_link_all.add_argument("--top_k", type=int, default=5)
    p_link_all.add_argument("--content_chars", type=int, default=512)
    p_link_all.add_argument("--min_content_len", type=int, default=0)
    p_link_all.add_argument("--anchor_from_sentence", action="store_true")
    p_link_all.add_argument("--min_score", type=float, default=0.0)
    p_link_all.add_argument("--anchor_max_len", type=int, default=120)
    p_link_all.add_argument("--output_format", choices=["csv","jsonl","parquet"], default="csv")
    p_link_all.add_argument("--overwrite", action="store_true")
    p_link_all.add_argument("--verbose", action="store_true")

    p_link_one = link_sub.add_parser("target")
    p_link_one.add_argument("--pages", required=True)
    p_link_one.add_argument("--output", required=True)
    p_link_one.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_link_one.add_argument("--top_k", type=int, default=5)
    p_link_one.add_argument("--target_url", default=None)
    p_link_one.add_argument("--target_topic", default=None)
    p_link_one.add_argument("--content_chars", type=int, default=512)
    p_link_one.add_argument("--min_content_len", type=int, default=0)
    p_link_one.add_argument("--anchor_from_sentence", action="store_true")
    p_link_one.add_argument("--min_score", type=float, default=0.0)
    p_link_one.add_argument("--anchor_max_len", type=int, default=120)
    p_link_one.add_argument("--output_format", choices=["csv","jsonl","parquet"], default="csv")
    p_link_one.add_argument("--overwrite", action="store_true")
    p_link_one.add_argument("--verbose", action="store_true")

    # Sitemaps
    p_sm = sub.add_parser("sitemaps", help="Sitemap utilities")
    sm_sub = p_sm.add_subparsers(dest="sm_cmd", required=True)

    p_sm_gen = sm_sub.add_parser("generate")
    p_sm_gen.add_argument("--pages", required=True)
    p_sm_gen.add_argument("--out_dir", default="public/sitemaps")
    p_sm_gen.add_argument("--max_urls", type=int, default=50000)
    p_sm_gen.add_argument("--base_index_url", default=None)
    p_sm_gen.add_argument("--gzip", action="store_true")
    p_sm_gen.add_argument("--lastmod_datetime", action="store_true")
    p_sm_gen.add_argument("--ensure_trailing_slash", action="store_true")

    p_sm_val = sm_sub.add_parser("validate")
    p_sm_val.add_argument("--path", required=True)

    p_sm_cmp = sm_sub.add_parser("compare")
    p_sm_cmp.add_argument("--sitemaps", nargs="+", required=True)
    p_sm_cmp.add_argument("--pages", required=True)
    p_sm_cmp.add_argument("--expand_index", action="store_true")
    p_sm_cmp.add_argument("--timeout", type=int, default=10)
    p_sm_cmp.add_argument("--orphans_out", default=None)
    p_sm_cmp.add_argument("--rogues_out", default=None)

    # PSI (delegate to its own parser for simplicity)
    p_psi = sub.add_parser("psi", help="Core Web Vitals snapshot (PSI API)")
    # We will forward all args to psi_snapshot.main to avoid duplication

    args, unknown = parser.parse_known_args()

    if args.cmd == "intent":
        classify_intent(args.input, args.output, args.model)
    elif args.cmd == "cluster":
        cluster_keywords(args.input, args.output, args.model, args.clusters)
    elif args.cmd == "meta":
        generate_meta(args.input, args.output, args.model)
    elif args.cmd == "ner":
        extract_entities(args.input, args.output, args.model)
    elif args.cmd == "links":
        if args.mode == "batch":
            df = suggest_for_all(args.pages, args.output, args.model, args.top_k, args.content_chars, args.min_content_len, args.anchor_from_sentence, args.min_score, args.anchor_max_len)
            from src.utils import write_dataframe
            write_dataframe(df, args.output, fmt=args.output_format, overwrite=args.overwrite, float_format='%.6f')
            print(f"Saved: {args.output}")
        else:
            df = suggest_for_target(args.pages, args.output, args.target_url, args.target_topic, args.model, args.top_k, args.content_chars, args.min_content_len, args.anchor_from_sentence, args.min_score, args.anchor_max_len)
            from src.utils import write_dataframe
            write_dataframe(df, args.output, fmt=args.output_format, overwrite=args.overwrite, float_format='%.6f')
            print(f"Saved: {args.output}")
    elif args.cmd == "sitemaps":
        if args.sm_cmd == "generate":
            generate_sitemaps(args.pages, args.out_dir, args.max_urls, args.base_index_url, args.gzip, args.lastmod_datetime, args.ensure_trailing_slash)
        elif args.sm_cmd == "validate":
            path, ok, msg = validate_sitemap_file(args.path)
            print(f"{path}\t{ok}\t{msg}")
        else:
            orphan_df, rogue_df = compare_sitemap_vs_urls(args.sitemaps, args.pages, args.expand_index, args.timeout)
            print(f"Orphans: {len(orphan_df)}\tRogues: {len(rogue_df)}")
            if args.orphans_out:
                orphan_df.to_csv(args.orphans_out, index=False)
                print(f"Saved orphans: {args.orphans_out}")
            if args.rogues_out:
                rogue_df.to_csv(args.rogues_out, index=False)
                print(f"Saved rogues: {args.rogues_out}")
    elif args.cmd == "psi":
        # Delegate to psi module for full arg handling
        psi_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
