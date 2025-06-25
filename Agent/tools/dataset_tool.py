from __future__ import annotations
from typing import List, Dict, Any, Optional
import pandas as pd
import re



class DatasetTool:
    """
    Tool for querying the main dataset (110k rows) using pandas operations.
    Starts with basic structured queries and can be extended later.
    """
    
    def __init__(self, dataset_df: pd.DataFrame, title_index=None):
        self.df = dataset_df
        self.title_index = title_index
        
        # Create indices for faster lookups
        self._create_indices()
        
        # Print column info for debugging
        if self.df is not None:
            print(f"DatasetTool initialized with {len(self.df)} rows")
            print(f"Columns: {list(self.df.columns)}")
    
    def _create_indices(self):
        """Create indices for common query patterns"""
        if self.df is not None:
            # Index by CELEX IDs if they exist
            if 'CELEX_FROM' in self.df.columns:
                self.df.set_index('CELEX_FROM', drop=False, inplace=False)
    
    def search_by_celex(self, celex_id: str, max_results: int = 10) -> List[pd.Series]:
        """Search for records by CELEX ID"""
        if self.df is None or celex_id is None:
            return []
        
        try:
            celex_clean = str(celex_id).strip()
            
            # Check if columns exist
            from_col = 'CELEX_FROM' if 'CELEX_FROM' in self.df.columns else None
            to_col = 'CELEX_TO' if 'CELEX_TO' in self.df.columns else None
            
            if not from_col and not to_col:
                print(f"Warning: No CELEX columns found in dataset. Available columns: {list(self.df.columns)}")
                return []
            
            # Search in available columns
            mask = pd.Series([False] * len(self.df))
            
            if from_col:
                mask_from = self.df[from_col].astype(str).str.contains(celex_clean, na=False, case=False)
                mask = mask | mask_from
            
            if to_col:
                mask_to = self.df[to_col].astype(str).str.contains(celex_clean, na=False, case=False)
                mask = mask | mask_to
            
            results = self.df[mask].head(max_results)
            return [row for _, row in results.iterrows()]
            
        except Exception as e:
            print(f"Error searching by CELEX {celex_id}: {e}")
            return []
    
    def search_by_case_title(self, case_title: str, max_results: int = 10) -> List[pd.Series]:
        """Search for records by case title"""
        if self.df is None or not case_title:
            return []
        
        try:
            title_clean = str(case_title).strip()
            
            # Check for title columns
            from_col = 'TITLE_FROM' if 'TITLE_FROM' in self.df.columns else None
            to_col = 'TITLE_TO' if 'TITLE_TO' in self.df.columns else None
            
            if not from_col and not to_col:
                print(f"Warning: No TITLE columns found in dataset. Available columns: {list(self.df.columns)}")
                return []
            
            # Search in available columns
            mask = pd.Series([False] * len(self.df))
            
            if from_col:
                mask_from = self.df[from_col].astype(str).str.contains(title_clean, na=False, case=False)
                mask = mask | mask_from
            
            if to_col:
                mask_to = self.df[to_col].astype(str).str.contains(title_clean, na=False, case=False)
                mask = mask | mask_to
            
            results = self.df[mask].head(max_results)
            return [row for _, row in results.iterrows()]
            
        except Exception as e:
            print(f"Error searching by case title {case_title}: {e}")
            return []
    
    def search_by_keywords(self, keywords: List[str], max_results: int = 10) -> List[pd.Series]:
        """Search for records containing keywords"""
        if self.df is None or not keywords:
            return []
        
        try:
            pattern = '|'.join([re.escape(kw) for kw in keywords])
            
            # Check for text columns
            text_columns = [col for col in self.df.columns if 'TEXT' in col.upper()]
            
            if not text_columns:
                print(f"Warning: No TEXT columns found in dataset. Available columns: {list(self.df.columns)}")
                return []
            
            # Search in all text columns
            mask = pd.Series([False] * len(self.df))
            
            for col in text_columns:
                col_mask = self.df[col].astype(str).str.contains(pattern, na=False, case=False)
                mask = mask | col_mask
            
            results = self.df[mask].head(max_results)
            return [row for _, row in results.iterrows()]
            
        except Exception as e:
            print(f"Error searching by keywords {keywords}: {e}")
            return []
    
    def get_related_cases(self, celex_id: str, max_results: int = 5) -> List[pd.Series]:
        """Get cases that cite or are cited by the given case"""
        if self.df is None or not celex_id:
            return []
        
        try:
            celex_clean = str(celex_id).strip()
            
            # Find cases where this case is cited (appears in TO columns)
            citing_mask = self.df['CELEX_TO'].astype(str).str.contains(celex_clean, na=False, case=False)
            
            # Find cases that this case cites (appears in FROM columns)
            cited_mask = self.df['CELEX_FROM'].astype(str).str.contains(celex_clean, na=False, case=False)
            
            results = self.df[citing_mask | cited_mask].head(max_results)
            return [row for _, row in results.iterrows()]
            
        except Exception as e:
            print(f"Error finding related cases for {celex_id}: {e}")
            return []
