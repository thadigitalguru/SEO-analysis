import os
import tempfile
from datetime import datetime, timezone
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

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def atomic_write(data: bytes, final_path: str):
    directory = os.path.dirname(final_path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, final_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def write_dataframe(df: pd.DataFrame, output_path: str, fmt: str = "csv", overwrite: bool = True, float_format: str = None):
    fmt = fmt.lower()
    if not overwrite and os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    if fmt == "csv":
        data = df.to_csv(index=False, encoding="utf-8", float_format=float_format).encode("utf-8")
        atomic_write(data, output_path)
    elif fmt == "jsonl":
        data = df.to_json(orient="records", lines=True).encode("utf-8")
        atomic_write(data, output_path)
    elif fmt == "parquet":
        # write to temp then move
        directory = os.path.dirname(output_path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=directory)
        os.close(fd)
        try:
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, output_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
    else:
        raise ValueError(f"Unsupported output format: {fmt}")