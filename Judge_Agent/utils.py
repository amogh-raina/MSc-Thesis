# Utility functions for Judge Agent
# File loading and data processing utilities

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def load_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load CSV/Excel file and convert to structured trios for sequential processing.
    This function is called externally before graph creation.
    
    Expected columns: question, reference_answer, generated_answer
    
    Args:
        file_path: Path to CSV or Excel file
        
    Returns:
        List of trio dictionaries with question, reference_answer, generated_answer
        
    Raises:
        ValueError: If file format is unsupported, columns are missing, or processing fails
    """
    try:
        # Load file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}. Expected .csv or .xlsx")
        
        # Validate required columns
        required_columns = ['question', 'reference_answer', 'generated_answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            available_columns = list(df.columns)
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {available_columns}"
            )
        
        # Process rows into trios
        trios = []
        skipped_rows = 0
        
        for index, row in df.iterrows():
            # Clean and validate data
            question = str(row['question']).strip()
            reference_answer = str(row['reference_answer']).strip()
            generated_answer = str(row['generated_answer']).strip()
            
            # Skip empty or invalid rows
            if not question or not reference_answer or not generated_answer:
                skipped_rows += 1
                continue
            
            # Skip rows with 'nan' values (from pandas)
            if question.lower() == 'nan' or reference_answer.lower() == 'nan' or generated_answer.lower() == 'nan':
                skipped_rows += 1
                continue
                
            trio = {
                'question': question,
                'reference_answer': reference_answer,
                'generated_answer': generated_answer
            }
            trios.append(trio)
        
        if skipped_rows > 0:
            print(f"⚠️  Skipped {skipped_rows} empty or invalid rows")
            
        if not trios:
            raise ValueError("No valid trios found in file after processing")
            
        return trios
        
    except Exception as e:
        raise ValueError(f"Error loading file '{file_path}': {str(e)}") 