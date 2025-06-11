"""
CSV *or* JSON → Parquet-cached DataFrame loader.
Path is injected by the caller (usually rag_app.py).
"""
from pathlib import Path
import pandas as pd

def load_df(source_path: str | Path) -> pd.DataFrame:
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(src)

    pq_path = src.with_suffix(".parquet")

    # ① use cached Parquet if it is newer than the source
    if pq_path.exists() and pq_path.stat().st_mtime >= src.stat().st_mtime:
        return pd.read_parquet(pq_path)

    # ② ingest CSV or JSON; **no other formats are assumed**
    if src.suffix.lower() == ".csv":
        df = pd.read_csv(src)
    elif src.suffix.lower() in {".json", ".jsonl"}:
        df = pd.read_json(src, lines=src.suffix.lower() == ".jsonl")
    else:
        raise ValueError(f"Unsupported extension: {src.suffix}")

    df.to_parquet(pq_path, compression="snappy")
    return df