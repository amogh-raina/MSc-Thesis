"""Fast fuzzy look-up from case titles to CELEX IDs."""

from __future__ import annotations
import re
from collections import defaultdict
from typing import Dict, List, FrozenSet, Iterable


class TitleIndex:
    # ------------------------------------------------------------------ #
    # construction / canonicalisation
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        #   canonical-token-set → [celex, …]
        self._token2celex: Dict[FrozenSet[str], List[str]] = defaultdict(list)
        #   celex → original case-title (first one seen wins)
        self._celex2title: Dict[str, str] = {}

    @staticmethod
    def _canon(text: str) -> FrozenSet[str]:
        """Lower-case, strip punctuation, return *set* of tokens."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return frozenset(cleaned.split())

    # ------------------------------------------------------------------ #
    # public api
    # ------------------------------------------------------------------ #
    def add(self, title: str, celex: str) -> None:
        """Register one (title, celex) pair."""
        if celex not in self._celex2title:
            self._celex2title[celex] = title
        self._token2celex[self._canon(title)].append(celex)

    @classmethod
    def from_df(cls, df) -> "TitleIndex":
        """
        Build index from a *paragraph-to-paragraph* dataframe.

        Recognised column schemes
        -------------------------
        1.  TITLE_FROM , CELEX_FROM , TITLE_TO , CELEX_TO   (your dataset)
        2.  citing_case_title / citing_celex
            cited_case_title  / cited_celex   (older variant)
        """
        idx = cls()

        if {"TITLE_FROM", "CELEX_FROM", "TITLE_TO", "CELEX_TO"}.issubset(df.columns):
            column_pairs = [
                ("TITLE_FROM", "CELEX_FROM"),
                ("TITLE_TO",   "CELEX_TO"),
            ]
        else:  # fallback to earlier lower-case schema
            column_pairs = [
                ("citing_case_title", "citing_celex"),
                ("cited_case_title",  "cited_celex"),
            ]

        for row in df.itertuples(index=False):
            for t_col, c_col in column_pairs:
                idx.add(getattr(row, t_col), getattr(row, c_col))

        return idx

    def lookup(self, mention: str, thresh: float = 0.7) -> List[str]:
        """
        Return the list of CELEX IDs whose *token-set Jaccard similarity*
        with the mention is ≥ `thresh`.

        Example
        -------
        >>> idx.lookup("Giersch v Luxembourg")
        ['62013CJ0020']
        """
        tokens = self._canon(mention)
        hits: List[str] = []
        for key, celexes in self._token2celex.items():
            if len(tokens & key) / len(tokens | key) >= thresh:
                hits.extend(celexes)
        return hits

    def get_title(self, celex: str) -> str | None:
        """Reverse-lookup: CELEX → canonical case title (if known)."""
        return self._celex2title.get(celex)
