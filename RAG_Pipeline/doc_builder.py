"""
Convert a P2P dataframe into LangChain Documents with enhanced grounding.

Document Building Strategy Comparison:
=====================================

1. build_paragraph_docs() - Full Citation Format
   - Format: (#CELEX:PARA) [TITLE] TEXT  
   - Creates 2 docs per row (citing + cited)
   - Rich metadata + embedded identifiers
   - Best for: Citation analysis, full legal context

2. build_case_paragraph_docs() - Case-Centric Format
   - Format: [TITLE]\nTEXT
   - Creates 1 doc per unique (celex, para) pair
   - CELEX only in metadata 
   - Best for: General legal Q&A, content search

3. build_optimized_legal_docs() - Flexible Hybrid (RECOMMENDED)
   - Format: Configurable ([TITLE] (#CELEX:PARA) TEXT)
   - Creates 1 doc per unique (celex, para) pair
   - CELEX embedded in content + rich metadata
   - Best for: RAG systems needing both search and citation

Choose based on your use case:
- Need exact citations in answers? → build_optimized_legal_docs with CELEX embedding
- Pure semantic search? → build_case_paragraph_docs  
- Full citation analysis? → build_paragraph_docs

- Always uses the 'standard' format: (#CELEX:PARA) [TITLE] TEXT
- Filters empty rows to prevent bloat
- Embeds identifiers directly in page_content for stronger LLM grounding
- Two docs per citation pair (one citing, one cited) for crisp vectors
- Column names are FULLY parametric via `citing_cols` / `cited_cols`
- Includes rich metadata for each Document
- Uses logging for reporting
"""

from __future__ import annotations
from uuid import uuid4
from typing import Dict, List
import math
import pandas as pd
import logging
from langchain.schema import Document

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------- #
# Default column mappings
# --------------------------------------------------------------------- #
_DEFAULT_CITING = {
    "text": "TEXT_FROM",
    "celex": "CELEX_FROM",
    "para": "NUMBER_FROM",
    "title": "TITLE_FROM",
    "date": "DATE_FROM",
}

_DEFAULT_CITED = {
    "text": "TEXT_TO",
    "celex": "CELEX_TO",
    "para": "NUMBER_TO",
    "title": "TITLE_TO",
    "date": "DATE_TO",
}

# --------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------- #
def _row_val(row, colname: str | None) -> str:
    """Return safe str value; empty string if NaN/None/missing."""
    if colname is None:
        return ""
    try:
        val = getattr(row, colname)
    except AttributeError:
        return ""
    if val is None or (isinstance(val, float) and math.isnan(val)) or pd.isna(val):
        return ""
    return str(val).strip()

def _format_page_content(
    text: str,
    celex: str,
    para: str,
    title: str,
) -> str:
    """
    Standard format: (#CELEX:PARA) [TITLE] TEXT
    """
    if not text.strip():
        return text
    identifier_parts = []
    if celex and para:
        identifier_parts.append(f"#{celex}:{para}")
    elif celex:
        identifier_parts.append(f"#{celex}")
    if title:
        identifier_parts.append(f"[{title}]")
    if identifier_parts:
        prefix = "(" + ") (".join(identifier_parts) + ")"
        return f"{prefix} {text}"
    else:
        return text

def _filter_empty_rows(
    df: pd.DataFrame,
    citing_text_col: str,
    cited_text_col: str
) -> pd.DataFrame:
    """Filter out rows where both citing and cited text are empty/NaN."""
    return df.dropna(subset=[citing_text_col, cited_text_col], how="all")

# --------------------------------------------------------------------- #
# Main factory function
# --------------------------------------------------------------------- #
def build_paragraph_docs(
    df: pd.DataFrame,
    citing_cols: Dict[str, str] | None = None,
    cited_cols: Dict[str, str] | None = None,
    filter_empty: bool = True,
    include_metadata: bool = True
) -> List[Document]:
    """
    Build LangChain Documents from P2P citation dataframe with enhanced grounding.
    Always uses the standard format: (#CELEX:PARA) [TITLE] TEXT

    Parameters
    ----------
    df : pd.DataFrame
        Source P2P dataframe with citation pairs.
    citing_cols : dict, optional
        Column mapping for citing side. Required keys: 'text', 'celex', 'para', 'title'
        Optional keys: 'date', 'subject'
    cited_cols : dict, optional
        Column mapping for cited side. Same structure as citing_cols.
    filter_empty : bool, default True
        Whether to filter out rows with empty text on both sides.
    include_metadata : bool, default True
        Whether to include rich metadata in Document objects.
    Returns
    -------
    List[Document]
        Two documents per row: one for citing paragraph, one for cited paragraph.
    """
    citing_cols = {**_DEFAULT_CITING, **(citing_cols or {})}
    cited_cols = {**_DEFAULT_CITED, **(cited_cols or {})}
    if filter_empty:
        original_count = len(df)
        df = _filter_empty_rows(df, citing_cols["text"], cited_cols["text"])
        filtered_count = len(df)
        if original_count > filtered_count:
            logger.info(f"Filtered {original_count - filtered_count} empty rows, processing {filtered_count} citation pairs")
    docs: List[Document] = []
    for row in df.itertuples(index=False):
        pair_id = str(uuid4())
        # Extract values for both sides
        citing_text = _row_val(row, citing_cols["text"])
        citing_celex = _row_val(row, citing_cols["celex"])
        citing_para = _row_val(row, citing_cols["para"])
        citing_title = _row_val(row, citing_cols["title"])
        citing_date = _row_val(row, citing_cols.get("date"))
        citing_subject = _row_val(row, citing_cols.get("subject"))
        cited_text = _row_val(row, cited_cols["text"])
        cited_celex = _row_val(row, cited_cols["celex"])
        cited_para = _row_val(row, cited_cols["para"])
        cited_title = _row_val(row, cited_cols["title"])
        cited_date = _row_val(row, cited_cols.get("date"))
        cited_subject = _row_val(row, cited_cols.get("subject"))
        # Skip if both texts are empty (additional safety check)
        if not citing_text.strip() and not cited_text.strip():
            continue
        # Format page content with embedded identifiers
        citing_content = _format_page_content(
            citing_text, citing_celex, citing_para, citing_title
        )
        cited_content = _format_page_content(
            cited_text, cited_celex, cited_para, cited_title
        )
        # Build metadata
        base_citing_metadata = {
            "role": "citing",
            "celex": citing_celex,
            "para_no": citing_para,
            "case_title": citing_title,
            "linked_celex": cited_celex,
            "linked_para_no": cited_para,
            "pair_id": pair_id,
            "date": citing_date,
            "subject_matter": citing_subject,
        }
        base_cited_metadata = {
            "role": "cited",
            "celex": cited_celex,
            "para_no": cited_para,
            "case_title": cited_title,
            "linked_celex": citing_celex,
            "linked_para_no": citing_para,
            "pair_id": pair_id,
            "date": cited_date,
            "subject_matter": cited_subject,
        }
        # Create citing document
        if citing_content.strip():
            docs.append(Document(
                page_content=citing_content,
                metadata=base_citing_metadata
            ))
        # Create cited document
        if cited_content.strip():
            docs.append(Document(
                page_content=cited_content,
                metadata=base_cited_metadata
            ))
    logger.info(f"Generated {len(docs)} documents from {len(df)} citation pairs")
    return docs

# --------------------------------------------------------------------- #
# Convenience function for standard ECJ P2P dataset
# --------------------------------------------------------------------- #
def build_docs_for_ecj_dataset(
    df: pd.DataFrame,
    filter_empty: bool = True
) -> List[Document]:
    """
    Convenience function for the standard ECJ P2P dataset format.
    Assumes columns: CELEX_FROM/TO, TITLE_FROM/TO, NUMBER_FROM/TO, DATE_FROM/TO, TEXT_FROM/TO
    """
    return build_paragraph_docs(
        df=df,
        citing_cols=_DEFAULT_CITING,
        cited_cols=_DEFAULT_CITED,
        filter_empty=filter_empty
    )

def build_case_paragraph_docs(
    df: pd.DataFrame,
    citing_cols: Dict[str, str] | None = None,
    cited_cols: Dict[str, str] | None = None,
    filter_empty: bool = True,
) -> list[Document]:
    """
    Build one Document per unique (celex, para_no) pair for both citing and cited sides.
    Content: [case_title]\nPARAGRAPH_TEXT
    Metadata: celex, date, para_no, case_title
    """
    citing_cols = {**_DEFAULT_CITING, **(citing_cols or {})}
    cited_cols = {**_DEFAULT_CITED, **(cited_cols or {})}
    seen = set()
    docs = []
    for row in df.itertuples(index=False):
        # Citing side
        celex = _row_val(row, citing_cols["celex"])
        para = _row_val(row, citing_cols["para"])
        title = _row_val(row, citing_cols["title"])
        date = _row_val(row, citing_cols.get("date"))
        text = _row_val(row, citing_cols["text"])
        key = (celex, para)
        if celex and para and key not in seen and text.strip():
            seen.add(key)
            content = f"[{title}]\n{text.strip()}"
            docs.append(Document(
                page_content=content,
                metadata={
                    "celex": celex,
                    "para_no": para,
                    "case_title": title,
                    "date": date,
                }
            ))
        # Cited side
        celex = _row_val(row, cited_cols["celex"])
        para = _row_val(row, cited_cols["para"])
        title = _row_val(row, cited_cols["title"])
        date = _row_val(row, cited_cols.get("date"))
        text = _row_val(row, cited_cols["text"])
        key = (celex, para)
        if celex and para and key not in seen and text.strip():
            seen.add(key)
            content = f"[{title}]\n{text.strip()}"
            docs.append(Document(
                page_content=content,
                metadata={
                    "celex": celex,
                    "para_no": para,
                    "case_title": title,
                    "date": date,
                }
            ))
    return docs

# --------------------------------------------------------------------- #
# Optimized document building for legal RAG
# --------------------------------------------------------------------- #
def build_optimized_legal_docs(
    df: pd.DataFrame,
    citing_cols: Dict[str, str] | None = None,
    cited_cols: Dict[str, str] | None = None,
    filter_empty: bool = True,
    include_celex_in_content: bool = True,
    format_style: str = "legal_standard"  # "legal_standard", "enhanced", "minimal"
) -> List[Document]:
    """
    Build optimized documents for legal RAG with CELEX embedding options.
    
    This function provides the best balance of:
    - Semantic searchability (embeddings work well)
    - Legal precision (CELEX IDs for exact citation)
    - Content richness (full context available)
    - Deduplication (one doc per unique paragraph)
    
    Parameters
    ----------
    df : pd.DataFrame
        Source P2P dataframe with citation pairs.
    citing_cols : dict, optional
        Column mapping for citing side. 
    cited_cols : dict, optional
        Column mapping for cited side.
    filter_empty : bool, default True
        Whether to filter out rows with empty text.
    include_celex_in_content : bool, default True
        Whether to embed CELEX IDs in the page content for LLM access.
    format_style : str, default "legal_standard"
        Document formatting style:
        - "legal_standard": [Title] (#CELEX:PARA) Text
        - "enhanced": (#CELEX:PARA) [Title] Text  
        - "minimal": [Title] Text (CELEX in metadata only)
    
    Returns
    -------
    List[Document]
        One document per unique (CELEX, paragraph) combination.
    """
    citing_cols = {**_DEFAULT_CITING, **(citing_cols or {})}
    cited_cols = {**_DEFAULT_CITED, **(cited_cols or {})}
    
    if filter_empty:
        original_count = len(df)
        df = _filter_empty_rows(df, citing_cols["text"], cited_cols["text"])
        filtered_count = len(df)
        if original_count > filtered_count:
            logger.info(f"Filtered {original_count - filtered_count} empty rows, processing {filtered_count} citation pairs")
    
    seen = set()
    docs = []
    
    for row in df.itertuples(index=False):
        # Process both citing and cited sides
        for side, cols in [("citing", citing_cols), ("cited", cited_cols)]:
            celex = _row_val(row, cols["celex"])
            para = _row_val(row, cols["para"])
            title = _row_val(row, cols["title"])
            date = _row_val(row, cols.get("date"))
            text = _row_val(row, cols["text"])
            
            # Create unique key and skip if already processed
            # Key is just (celex, para) - we want one doc per unique paragraph regardless of role
            key = (celex, para)
            if celex and para and key not in seen and text.strip():
                seen.add(key)
                
                # Format content based on style
                if format_style == "legal_standard" and include_celex_in_content:
                    content = f"[{title}] (#{celex}:{para})\n{text.strip()}"
                elif format_style == "enhanced" and include_celex_in_content:
                    content = f"(#{celex}:{para}) [{title}]\n{text.strip()}"
                else:  # minimal or no celex in content
                    content = f"[{title}]\n{text.strip()}"
                
                # Rich metadata for all approaches
                metadata = {
                    "celex": celex,
                    "para_no": para,
                    "case_title": title,
                    "date": date,
                    "role": side,
                    "content_format": format_style,
                    "has_celex_in_content": include_celex_in_content
                }
                
                docs.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
    
    logger.info(f"Generated {len(docs)} optimized legal documents from {len(df)} citation pairs")
    return docs
