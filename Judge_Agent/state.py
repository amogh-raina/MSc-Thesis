# State schema definition for Judge Agent 
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict


class JudgeState(TypedDict):
    """
    Simplified state schema for Judge Agent focused on agent execution.
    File processing is handled externally before graph creation.
    """
    # Current trio being processed
    current_trio: Dict[str, str]  # {question, reference_answer, generated_answer}
    
    # All trios from the file (pre-loaded externally)
    all_trios: List[Dict[str, str]]
    
    # Current trio index (for tracking progress)
    current_index: int
    
    # Optional search results if agent decides to search
    search_results: Optional[str]
    
    # Final evaluation scores for current trio
    current_scores: Optional[Dict[str, Any]]
    
    # All completed evaluations
    all_evaluations: List[Dict[str, Any]]
    
    # Processing status
    is_complete: bool