import pandas as pd
from dotenv import load_dotenv
from typing import List

def load_env():
    load_dotenv()

def read_keywords_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "keyword" not in df.columns:
        raise ValueError("CSV must include a 'keyword' column.")
    return df

def read_serp_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def ensure_cols(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"Saved: {path}")