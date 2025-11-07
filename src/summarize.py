"""
Summarize long content (e.g., competitor pages) into brief notes.
"""
import argparse
from transformers import pipeline

def summarize_text(text, model_name="sshleifer/distilbart-cnn-12-6", max_new_tokens=128):
    summarizer = pipeline("summarization", model=model_name)
    t = text.strip()
    if len(t) > 3000:
        t = t[:3000]
    out = summarizer(t, max_new_tokens=max_new_tokens)[0]["summary_text"]
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", type=str, default="Running shoes are designed with different foams...")
    parser.add_argument("--model", type=str, default="sshleifer/distilbart-cnn-12-6")
    args = parser.parse_args()
    print(summarize_text(args.input_text, model_name=args.model))