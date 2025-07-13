"""
GraphRAG Prompt Template
========================

Simple prompt template for GraphRAG that handles context from both:
- Vector search results (case content) 
- Graph expansion results (related content via relationships)

Reuses the same EU law citation format as RAG_Prompt.py
"""

from langchain.prompts import PromptTemplate

# Alternative: Just reuse RAG_Prompt.py directly
def _reuse_rag_prompt(style="detailed"):
    """
    Alternative function that imports and reuses RAG_Prompt.py template directly
    since it already has perfect EU law citation instructions
    """
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent
    rag_pipeline_path = project_root / "RAG_Pipeline"
    sys.path.insert(0, str(rag_pipeline_path))
        
    from RAG_Pipeline.RAG_Prompt import _rag_prompt
    return _rag_prompt(style)


# def prompt_chain(style="detailed"):
#     """
#     Create GraphRAG prompt chain for the UI interface
    
#     Args:
#         style: "detailed" or "concise"
        
#     Returns:
#         PromptTemplate configured for GraphRAG context + question input
#     """
#     return _graphrag_prompt(style)


if __name__ == "__main__":
    # print("GraphRAG Prompt Template - Simple Version")
    # print("=" * 50)
    # print("\n📝 Using GraphRAG prompt:")
    # graphrag_template = _graphrag_prompt("detailed")
    # print(f"Input variables: {graphrag_template.input_variables}")
    
    print("\n📝 Using reused RAG prompt:")
    reused_template = _reuse_rag_prompt("detailed")
    print(f"Input variables: {reused_template.input_variables}")
    
    print("\n✅ Both templates take 'context' and 'question' as inputs")
    print("💡 You can use either _graphrag_prompt() or _reuse_rag_prompt()")
