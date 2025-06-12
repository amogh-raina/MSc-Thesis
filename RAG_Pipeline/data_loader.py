"""
CSV *or* JSON → Parquet-cached DataFrame loader.
Path is injected by the caller (usually rag_app.py).
"""
from pathlib import Path
import pandas as pd
from langchain_community.document_loaders import CSVLoader, JSONLoader
from langchain.schema import Document

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

def load_documents(source_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV or JSON file into a DataFrame using LangChain's document loaders for consistent parsing.
    Returns a DataFrame with all columns from the file.
    """
    src = Path(source_path)
    if not src.exists():
        raise FileNotFoundError(src)
    if src.suffix.lower() == ".csv":
        loader = CSVLoader(file_path=str(src), encoding="utf-8", autodetect_encoding=True)
        docs = loader.load()
        # Convert list of Documents to DataFrame
        rows = []
        for doc in docs:
            # Each doc.page_content is a string of key: value pairs per row
            row = {}
            for line in doc.page_content.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    row[k.strip()] = v.strip()
            # Add metadata columns if present
            row.update(doc.metadata)
            rows.append(row)
        return pd.DataFrame(rows)
    elif src.suffix.lower() in {".json", ".jsonl"}:
        loader = JSONLoader(file_path=str(src))
        docs = loader.load()
        rows = []
        for doc in docs:
            # JSONLoader puts the whole object in page_content as a string, but metadata may have fields
            row = {}
            if isinstance(doc.page_content, str):
                try:
                    import json
                    row = json.loads(doc.page_content)
                except Exception:
                    pass
            row.update(doc.metadata)
            rows.append(row)
        return pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported extension: {src.suffix}")