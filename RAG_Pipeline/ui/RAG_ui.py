import sys
from pathlib import Path

# Add the project root and Main folder to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
main_folder = project_root / "Main"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(main_folder) not in sys.path:
    sys.path.insert(0, str(main_folder))

import streamlit as st

# --- Page Configuration ---
# This MUST be the first Streamlit command used in an app, and must only be
# set once. We put it here, right after the streamlit import.
st.set_page_config(
    page_title="RAG Legal Knowledge Evaluator",
    page_icon="🔍",
    layout="wide"
)

import os
import tempfile
import re
from io import BytesIO
import shutil
import asyncio
import json
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any
from langchain.schema import Document

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up tracing
os.environ["LANGCHAIN_TRACING_V2"] = "False"
os.environ["LANGCHAIN_PROJECT"] = "MSc_Thesis"
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Import shared components from Main folder
from Main.core.model_manager import ModelManager, EmbeddingManager
from Main.core.question_bank import QuestionBank
from Main.core.evaluator import LLMEvaluator
from Main.config.settings import *

# RAG Pipeline imports
from RAG_Pipeline.rag_pipeline import RAGPipeline
from RAG_Pipeline.variants.hybrid import HybridRAGPipeline, create_large_hybrid_dataset_pipeline
from RAG_Pipeline.title_index import TitleIndex
from RAG_Pipeline.Langchain_eval_framework import LangchainEvalPipeline
from RAG_Pipeline.RAGAS_eval_framework import RAGASEvalPipeline
from RAG_Pipeline.RAG_Prompt import _rag_prompt

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    st.title("🔍 RAG Legal Knowledge Evaluator")
    st.markdown("Evaluate RAG system performance on legal questions with document retrieval")
    st.markdown("---")
    
    # Initialize evaluator
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = LLMEvaluator()
        st.session_state.evaluation_history = []
        st.session_state.question_bank_ready = False
        st.session_state.embedding_enabled = False
    if "rag_evaluation_history" not in st.session_state:
        st.session_state.rag_evaluation_history = []

    # Handle fallback export request if direct call from sidebar failed
    if st.session_state.get("export_requested", False):
        export_format = st.session_state.get("export_format", "excel")
        if st.session_state.evaluation_history:
            try:
                export_results(st.session_state.evaluation_history, export_format)
                st.success("✅ Export completed successfully (fallback method)")
            except Exception as e:
                st.error(f"❌ Export failed even with fallback: {str(e)}")
        else:
            st.warning("⚠️ No evaluation results to export")
        
        # Clear the export request flag
        st.session_state.export_requested = False
        if "export_format" in st.session_state:
            del st.session_state.export_format

    # Sidebar configuration
    selected_llm_provider, selected_llm_model, response_type = sidebar_configuration()
    
    # Main interface
    if not st.session_state.question_bank_ready:
        st.info("👈 Please load the question bank from the sidebar to begin evaluation")
    elif st.session_state.rag_system is None:
        st.info("👈 Please upload dataset and build RAG system from the sidebar")
    else:
        if selected_llm_provider and selected_llm_model:
            tab1, tab2 = st.tabs([
                "🔍 Manual RAG Evaluation",
                "📊 Batch RAG Evaluation"
            ])
            
            with tab1:
                manual_rag_evaluation_interface(selected_llm_provider, selected_llm_model, response_type)
                
            with tab2:
                batch_rag_evaluation_interface(selected_llm_provider, selected_llm_model, response_type)
        else:
            st.warning("Please select LLM provider and model from sidebar")


def initialize_session_state():
    """Initialize session state variables similar to rag_app.py"""
    # RAG system state
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
        st.session_state.rag_retriever = None
        st.session_state.rag_title_idx = None

    # Evaluation state
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = LLMEvaluator()
        st.session_state.question_bank_ready = False
        st.session_state.embedding_enabled = False
    
    # Ensure evaluation histories are always initialized
    if "evaluation_history" not in st.session_state:
        st.session_state.evaluation_history = []
    if "rag_evaluation_history" not in st.session_state:
        st.session_state.rag_evaluation_history = []


def sidebar_configuration():
    """Sidebar configuration - main interface for settings"""
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data directory input
        st.subheader("📄 Question Database")
        data_dir = st.text_input(
            "Data Directory Path", 
            value="E:/Projects/MSc-Thesis/JSON Trial 1",
            help="Path to directory containing BEUL_EXAM_*.json files"
        )
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        
        llm_providers = ModelManager.get_available_llm_providers()
        
        if not llm_providers:
            st.error("❌ No LLM providers available. Please check your API keys.")
            return None, None, None
        
        selected_llm_provider = st.selectbox(
            "LLM Provider",
            options=list(llm_providers.keys()),
            help="Select the provider for your language model"
        )
        
        selected_llm_model = None
        if selected_llm_provider:
            selected_llm_model = st.selectbox(
                "LLM Model",
                options=llm_providers[selected_llm_provider]["models"],
                help="Select the specific model to use"
            )
        
        # Store in session state for agent use
        st.session_state.selected_llm_provider = selected_llm_provider
        st.session_state.selected_llm_model = selected_llm_model
        
        response_type = st.selectbox(
            "Response Type",
            ["detailed", "concise"],
            help="Choose whether to generate detailed or concise answers"
        )
        
        st.markdown("---")
        
        # Embedding Configuration
        st.subheader("🧠 Embedding Configuration")
        embedding_providers = EmbeddingManager.get_available_embedding_providers()

        # Manual evaluation embeddings
        use_embeddings_manual = st.checkbox(
            "Enable Embeddings (for manual evaluation only)",
            help="Embeddings are only used for manual question matching",
            key="enable_emb_manual"
        )

        # Semantic‑similarity embeddings
        use_embeddings_sem = st.checkbox(
            "Enable embeddings for semantic-similarity",
            help="Toggle to compute the semantic similarity metric",
            key="enable_emb_sem"
        )

        # Manual matching model selector
        selected_embedding_provider = None
        selected_embedding_model = None
        if use_embeddings_manual and embedding_providers:
            selected_embedding_provider = st.selectbox(
                "Embedding Provider (manual)",
                options=list(embedding_providers.keys()),
                key="emb_provider_manual"
            )
            if selected_embedding_provider:
                selected_embedding_model = st.selectbox(
                    "Embedding Model (manual)",
                    options=embedding_providers[selected_embedding_provider]["models"],
                    key="emb_model_manual"
                )

        # Semantic similarity model selector
        emb_provider_sem = None
        emb_model_name_sem = None
        if use_embeddings_sem and embedding_providers:
            emb_provider_sem = st.selectbox(
                "Embedding Provider (semantic Score)",
                options=list(embedding_providers.keys()),
                key="emb_provider_sem"
            )
            if emb_provider_sem:
                emb_model_name_sem = st.selectbox(
                    "Embedding Model (semantic Score)",
                    options=embedding_providers[emb_provider_sem]["models"],
                    key="emb_model_name_sem"
                )

        # RAG Configuration
        with st.sidebar.expander("📚 RAG Configuration"):
            st.markdown("Upload the **Paragraph-to-Paragraph** dataset (CSV or JSON) "
                        "and configure RAG-specific settings.")

            uploaded_db = st.file_uploader("Dataset file", type=["csv", "json", "jsonl"])
            chroma_dir  = st.text_input("Chroma directory", "./chroma_db")

            # RAG-specific embedding configuration
            st.markdown("**RAG Embedding Configuration**")
            use_rag_embeddings = st.checkbox(
                "Enable RAG Embeddings", 
                value=True,
                help="Enable embeddings for RAG document retrieval",
                key="enable_rag_embeddings"
            )
            
            rag_emb_provider = None
            rag_emb_model = None
            if use_rag_embeddings and embedding_providers:
                rag_emb_provider = st.selectbox(
                    "RAG Embedding Provider",
                    options=list(embedding_providers.keys()),
                    key="rag_emb_provider",
                    help="Select embedding provider for RAG retrieval"
                )
                if rag_emb_provider:
                    rag_emb_model = st.selectbox(
                        "RAG Embedding Model",
                        options=embedding_providers[rag_emb_provider]["models"],
                        key="rag_emb_model",
                        help="Select embedding model for RAG retrieval"
                    )
            
            # Retrieval settings
            st.markdown("**Retrieval Settings**")
            
            # Retrieval method selection
            retrieval_method = st.selectbox(
                "Retrieval Method",
                options=["Dense (Vector Only)", "Hybrid (Vector + BM25)"],
                index=0,
                help="Choose retrieval strategy:\n"
                     "• Dense: Semantic similarity search only\n"
                     "• Hybrid: Combines semantic + keyword matching for better legal citation recall",
                key="retrieval_method"
            )
            
            # Reranker selection (only show if Hybrid is selected)
            use_reranking = False
            rerank_type = "bge"
            rerank_model = "BAAI/bge-reranker-base"
            
            if retrieval_method == "Hybrid (Vector + BM25)":
                st.markdown("**Reranking Configuration**")
                st.info("💡 **Hybrid retrieval automatically uses 30% BM25 + 70% Vector weighting, optimized for legal Q&A**")
                
                reranker_choice = st.selectbox(
                    "Reranker Model",
                    options=["CrossEncoder (Local/Free)", "Jina Rerank (API Required)"],
                    index=0,
                    help="Choose reranking method:\n"
                         "• CrossEncoder: Free BGE models running locally\n"
                         "• Jina: API-based service (requires Jina API key)",
                    key="reranker_choice"
                )
                
                if reranker_choice == "CrossEncoder (Local/Free)":
                    use_reranking = True
                    rerank_type = "bge"
                    rerank_model = st.selectbox(
                        "BGE Model",
                        options=["BAAI/bge-reranker-base", "BAAI/bge-reranker-large", "BAAI/bge-reranker-v2-m3"],
                        index=0,
                        help="Choose BGE reranker model:\n"
                             "• base: Fast, good performance\n"
                             "• large: Better performance, more compute\n"
                             "• v2-m3: Latest model, best performance",
                        key="bge_model"
                    )
                elif reranker_choice == "Jina Rerank (API Required)":
                    use_reranking = True
                    rerank_type = "jina"
                    rerank_model = "jina-reranker-m0"
                
                # Dependency information (non-nested)
                st.markdown("**📦 Dependencies for Hybrid Retrieval:**")
                st.markdown("**Required packages:**")
                st.code("pip install rank-bm25")  # For BM25 retrieval
                
                st.info("💡 **Note:** Dependencies are automatically checked when building the pipeline.")
            
            k_value = st.number_input(
                "Number of documents to retrieve (k)",
                min_value=1,
                max_value=20,
                value=15,
                help="Number of relevant documents to retrieve for each query",
                key="rag_k_value"
            )

            col_build, col_reset = st.columns(2)
            if col_build.button("🔧 Build / Load", use_container_width=True, key="rag_build_btn"):
                if not uploaded_db:
                    st.error("Please upload a CSV/JSON first.")
                elif not use_rag_embeddings:
                    st.error("Please enable RAG embeddings.")
                elif not rag_emb_provider or not rag_emb_model:
                    st.error("Please select both RAG embedding provider and model.")
                else:
                    is_hybrid = retrieval_method == "Hybrid (Vector + BM25)"
                    _setup_rag(uploaded_db, chroma_dir, rag_emb_provider, rag_emb_model, 
                              k_value, is_hybrid, use_reranking, rerank_type, rerank_model)

            # Reset options
            col_reset1, col_reset2 = st.columns(2)
            if col_reset1.button("🗑 Reset", use_container_width=True, key="rag_reset_btn", 
                                help="Clear session state and delete files"):
                _reset_rag(chroma_dir)
            
            if col_reset2.button("🔄 Soft Reset", use_container_width=True, key="rag_soft_reset_btn",
                                help="Clear session state only (keep files)"):
                _soft_reset_rag()

        # LangChain RAG Evaluation Configuration
        with st.sidebar.expander("🔬 LangChain RAG Evaluation"):
            st.markdown("Configure LLM-as-a-Judge evaluation for RAG responses using LangChain framework")
            st.caption("💡 Evaluations use step-by-step reasoning internally. Only scores/boolean values are exported for zero-shot Q&A systems.")
            
            enable_rag_eval = st.checkbox(
                "Enable LangChain RAG Evaluation",
                key="enable_rag_eval",
                help="Enable 4-metric evaluation: correctness, relevance, groundedness, retrieval relevance"
            )
            
            if enable_rag_eval:
                # Evaluation model selection
                eval_provider = st.selectbox(
                    "Evaluation LLM Provider",
                    options=list(llm_providers.keys()) if llm_providers else [],
                    key="eval_llm_provider",
                    help="Select the LLM provider for evaluation"
                )
                
                eval_model = None
                if eval_provider:
                    eval_model = st.selectbox(
                        "Evaluation LLM Model",
                        options=llm_providers[eval_provider]["models"],
                        key="eval_llm_model",
                        help="Select the specific model for evaluation"
                    )
                
                # Display status without modifying session state here
                if eval_provider and eval_model:
                    st.success(f"✅ Evaluation LLM: {eval_provider}/{eval_model}")
                else:
                    st.warning("Please select both provider and model for evaluation")
            else:
                # Clear evaluation settings when disabled
                if "eval_llm_provider" in st.session_state:
                    del st.session_state.eval_llm_provider
                if "eval_llm_model" in st.session_state:
                    del st.session_state.eval_llm_model

        # RAGAS RAG Evaluation Configuration
        with st.sidebar.expander("📊 RAGAS RAG Evaluation"):
            st.markdown("Configure mathematically-grounded RAG evaluation using RAGAS metrics")
            st.caption("📚 Based on official RAGAS documentation patterns with proper LLM wrappers")
            
            enable_ragas_eval = st.checkbox(
                "Enable RAGAS RAG Evaluation",
                key="enable_ragas_eval",
                help="Enable 5-metric evaluation: context recall, context precision, faithfulness, factual correctness, answer accuracy"
            )
            
            if enable_ragas_eval:
                # Evaluation model selection
                ragas_provider = st.selectbox(
                    "RAGAS LLM Provider",
                    options=list(llm_providers.keys()) if llm_providers else [],
                    key="ragas_llm_provider",
                    help="Select the LLM provider for RAGAS evaluation"
                )
                
                ragas_model = None
                if ragas_provider:
                    ragas_model = st.selectbox(
                        "RAGAS LLM Model",
                        options=llm_providers[ragas_provider]["models"],
                        key="ragas_llm_model",
                        help="Select the specific model for RAGAS evaluation"
                    )
                

        # Question bank loading
        col1, col2 = st.columns(2)
        with col1:
            load_basic = st.button("Load Basic", help="Load without embeddings")
        with col2:
            load_enhanced = st.button("Load Enhanced", help="Load with embeddings")
        
        if load_basic or load_enhanced:
            with st.spinner("Loading question bank..."):
                if load_enhanced and use_embeddings_manual and selected_embedding_provider:
                    success = st.session_state.evaluator.setup_question_bank(
                        data_dir, selected_embedding_provider, selected_embedding_model
                    )
                    if success:
                        st.session_state.question_bank_ready = True
                        st.session_state.embedding_enabled = True
                        st.success("✅ Enhanced loading complete!")
                    else:
                        st.error("❌ Failed to load question bank")
                else:
                    success = st.session_state.evaluator.setup_question_bank(data_dir)
                    if success:
                        st.session_state.question_bank_ready = True
                        st.session_state.embedding_enabled = False
                        question_count = len(st.session_state.evaluator.question_bank.questions)
                        st.success(f"✅ Basic loading complete! Loaded {question_count} questions")
                    else:
                        st.error("❌ Failed to load question bank")
        
        st.markdown("---")
        
        # Export section
        st.subheader("📊 Export Results")
        
        total_evaluations = len(st.session_state.evaluation_history) if st.session_state.evaluation_history else 0
        
        # Session status info
        if total_evaluations > 0:
            latest_result = st.session_state.evaluation_history[-1]
            st.caption(f"Latest: {latest_result.get('timestamp', 'No timestamp')[:19]} - {latest_result.get('llm_provider', 'Unknown')}")
        
        if total_evaluations > 0:
            st.metric("Total Evaluations", total_evaluations)
            export_format = st.selectbox("Export Format", ["excel", "json"])
            
            if st.button("📥 Export Results"):
                # Call export directly but handle any context issues gracefully
                try:
                    export_results(st.session_state.evaluation_history, export_format)
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
                    # Fallback: Set flag for main loop to handle
                    st.session_state.export_requested = True
                    st.session_state.export_format = export_format
            
            # Add clear results option
            st.markdown("---")
            if st.button("🗑️ Clear All Results", help="Clear accumulated evaluation history"):
                st.session_state.evaluation_history = []
                st.session_state.rag_evaluation_history = []
                st.success("✅ All evaluation results cleared!")
                st.rerun()
        else:
            st.text("No evaluations to export yet")
            st.info("💡 **Workflow:** Run manual or batch evaluations to accumulate results. Export options appear immediately after results and are also available here.")
        
        return selected_llm_provider, selected_llm_model, response_type


def manual_rag_evaluation_interface(selected_llm_provider, selected_llm_model, response_type):
    """Manual RAG evaluation interface"""
    
    st.subheader("🔍 Manual RAG Evaluation")
    
    question = st.text_area(
        "Enter your legal question for RAG evaluation:",
        placeholder="e.g., What are the consequences of failing to implement EU directives?",
        height=100
    )
    
    # Reference answer handling
    reference_mode = st.radio(
        "Reference Answer Source:",
        ["🤖 Auto-find from database", "✏️ Provide manually"],
        horizontal=True
    )
    
    manual_reference_answer = None
    if reference_mode == "✏️ Provide manually":
        manual_reference_answer = st.text_area(
            "Manual Reference Answer:",
            placeholder="Provide the expected/correct answer for evaluation",
            height=150
        )
        
    if st.button("🔍 Generate RAG Answer & Evaluate", type="primary"):
        if not question.strip():
            st.error("❌ Please provide a question")
        elif reference_mode == "✏️ Provide manually" and not manual_reference_answer.strip():
            st.error("❌ Please provide a reference answer")
        else:
            if st.session_state.rag_retriever is None:
                st.error("❌ RAG system is not initialized. Please build the RAG system first.")
                return
            
            with st.spinner("🔍 Generating RAG answer and evaluating..."):
                # Generate RAG answer with smart retrieval
                llm = ModelManager.create_llm(selected_llm_provider, selected_llm_model)
                
                # Check if LLM creation was successful
                if llm is None:
                    st.error("❌ Failed to create LLM model. Please check the model availability and try again.")
                    st.info("💡 **Tip:** The selected model may be temporarily unavailable. Try a different model or check the service status.")
                    return
                
                # Get target k from RAG config
                target_k = st.session_state.get("rag_config", {}).get("k_value", 10)
                
                # Use smart retrieval to get properly deduplicated documents
                source_docs = smart_rag_retrieval(
                    retriever=st.session_state.rag_retriever,
                    query=question,
                    target_k=target_k,
                    expansion_factor=1.9  # Retrieve 90% more documents initially
                )
                
                # Create context from deduplicated documents
                context = "\n\n".join([doc.page_content for doc in source_docs])
                
                # Generate answer using LLM with retrieved context
                from langchain.prompts import PromptTemplate, ChatPromptTemplate
                prompt_template = _rag_prompt(response_type)
                
                # Handle both ChatPromptTemplate (CoT) and PromptTemplate (legacy)
                if isinstance(prompt_template, ChatPromptTemplate):
                    # For ChatPromptTemplate, format with context and question
                    formatted_prompt = prompt_template.format(question=question, context=context)
                elif hasattr(prompt_template, 'template'):
                    # For PromptTemplate, extract template string and format
                    template_str = prompt_template.template
                    formatted_prompt = template_str.format(question=question, context=context)
                else:
                    # Fallback
                    template_str = str(prompt_template)
                    formatted_prompt = template_str.format(question=question, context=context)
                answer = llm.invoke(formatted_prompt).content
                
                # Get reference answer
                if reference_mode == "🤖 Auto-find from database":
                    if st.session_state.embedding_enabled:
                        reference_info = st.session_state.evaluator.question_bank.find_reference_answer_embedding(question)
                        if reference_info and reference_info.get("question_data"):
                            reference_answer = reference_info["question_data"]["answer_text"]
                        elif reference_info and reference_info.get("answer_text"):
                            reference_answer = reference_info["answer_text"]
                        else:
                            reference_answer = None
                    else:
                        reference_answer = st.session_state.evaluator.question_bank.find_reference_answer(question)
                else:
                    reference_answer = manual_reference_answer
                
                if not reference_answer:
                    st.error("❌ No reference answer found for evaluation.")
                else:
                    # Evaluate the RAG answer
                    import asyncio
                    evaluation = asyncio.run(
                        st.session_state.evaluator.evaluate_single_response(answer, reference_answer)
                    )
                    
                    # Create result object
                    result = {
                        "question": question,
                        "generated_answer": answer,
                        "reference_answer": reference_answer,
                        "evaluation": evaluation,
                        "llm_provider": selected_llm_provider,
                        "llm_model": selected_llm_model,
                        "response_type": response_type,
                        "rag": True,
                        "timestamp": datetime.now().isoformat(),
                        "retrieved_context": context,
                        "source_docs": [doc.metadata for doc in source_docs],
                        "rag_config": st.session_state.get("rag_config", {}),
                    }
                    
                    # Run additional evaluations if enabled
                    result = run_langchain_rag_evaluation([result])[0] if st.session_state.get("enable_rag_eval", False) else result
                    result = run_ragas_rag_evaluation([result])[0] if st.session_state.get("enable_ragas_eval", False) else result
                    
                    # Store in both histories for backward compatibility and export functionality
                    st.session_state.rag_evaluation_history.append(result)
                    st.session_state.evaluation_history.append(result)
                    
                    # Confirm storage
                    total_stored = len(st.session_state.evaluation_history)
                    st.success(f"✅ RAG evaluation completed! ({total_stored} total results stored)")
                    
                    show_rag_evaluation_results(result)
                    
                    # Add immediate export options right after results
                    st.markdown("---")
                    st.subheader("📥 Export This Session")
                    st.caption(f"Export all {len(st.session_state.evaluation_history)} accumulated results")
                    
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        if st.button("📥 Download Excel Report", key="manual_excel_export"):
                            export_results(st.session_state.evaluation_history, "excel")
                    with col_exp2:
                        if st.button("📥 Download JSON Report", key="manual_json_export"):
                            export_results(st.session_state.evaluation_history, "json")

def show_evaluation_results(result):
    """Display detailed evaluation results"""
    
    st.markdown("---")
    st.subheader("📊 Detailed Evaluation Results")
    
    # Scores in columns
    col1, col2, col3, col4 = st.columns(4)
    evaluation = result["evaluation"]
    
    with col1:
        score = evaluation['bleu_score']
        color = "🟢" if score > 0.4 else "🟡" if score > 0.2 else "🔴"
        st.metric("BLEU Score", f"{score:.3f}")
        st.caption(f"{color} {'Excellent' if score > 0.4 else 'Good' if score > 0.2 else 'Needs improvement'}")

    with col2:
        score = evaluation['rouge_score']
        color = "🟢" if score > 0.4 else "🟡" if score > 0.2 else "🔴"
        st.metric("ROUGE Score", f"{score:.3f}")
        st.caption(f"{color} {'Excellent' if score > 0.4 else 'Good' if score > 0.2 else 'Needs improvement'}")

    with col3:
        score = evaluation['string_similarity_score']
        color = "🟢" if score > 0.6 else "🟡" if score > 0.4 else "🔴"
        st.metric("String Similarity", f"{score:.3f}")
        st.caption(f"{color} {'Excellent' if score > 0.6 else 'Good' if score > 0.4 else 'Needs improvement'}")

    with col4:
        if 'semantic_similarity_score' in evaluation:
            score = evaluation['semantic_similarity_score']
            color = "🟢" if score > 0.6 else "🟡" if score > 0.4 else "🔴"
            st.metric("Semantic Similarity", f"{score:.3f}")
            st.caption(f"{color} {'Excellent' if score > 0.6 else 'Good' if score > 0.4 else 'Needs improvement'}")
        else:
            st.metric("Semantic Similarity", "—")
    
    # Generated answer
    st.markdown("**🤖 Generated Answer:**")
    st.info(result["generated_answer"])
    
    # Reference lookup info
    if result.get("reference_lookup_info"):
        lookup_info = result["reference_lookup_info"]
        
        st.markdown("**🔍 Reference Matching Details:**")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write(f"**Method:** {lookup_info.get('matching_method', 'N/A')}")
            st.write(f"**Model:** {lookup_info.get('model_used', 'N/A')}")
        
        with col_b:
            if lookup_info.get('similarity_score'):
                st.write(f"**Match Similarity:** {lookup_info['similarity_score']:.3f}")
            if lookup_info.get('question_data'):
                matched_data = lookup_info['question_data']
                st.write(f"**Source:** {matched_data.get('year', 'N/A')}-Q{matched_data.get('question_number', 'N/A')}")
    
    # Model details
    st.markdown("**⚙️ Configuration:**")
    provider_info = f"{result['llm_provider']}/{result['llm_model']}"
    st.write(f"LLM: {provider_info}")
    st.write(f"Response Type: {result['response_type']}")
    st.write(f"Timestamp: {result['timestamp']}")


def batch_rag_evaluation_interface(selected_llm_provider, selected_llm_model, response_type):
    """Batch RAG evaluation interface"""
    
    st.subheader("📊 Batch RAG Evaluation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Evaluate multiple questions automatically using RAG system with database questions and answers")
        
        max_questions = st.number_input(
            "Maximum questions to evaluate:",
            min_value=1,
            max_value=100,
            value=10,
            help="Limit the number of questions for testing"
        )
        
        if st.button("🚀 Start Batch RAG Evaluation", type="primary"):
            with st.spinner("Running batch RAG evaluation..."):
                # Get questions from database
                questions = st.session_state.evaluator.question_bank.get_all_questions()
                questions = questions[:max_questions]
                results = []
                
                # Prepare for batch RAG evaluation with smart retrieval
                llm = ModelManager.create_llm(selected_llm_provider, selected_llm_model)
                
                # Check if LLM creation was successful
                if llm is None:
                    st.error("❌ Failed to create LLM model for batch evaluation. Please check the model availability and try again.")
                    st.info("💡 **Tip:** The selected model may be temporarily unavailable. Try a different model or check the service status.")
                    return
                
                # Get target k from RAG config
                target_k = st.session_state.get("rag_config", {}).get("k_value", 10)
                
                # Prepare prompt template
                from langchain.prompts import PromptTemplate, ChatPromptTemplate
                prompt_template = _rag_prompt(response_type)
                
                # Handle both ChatPromptTemplate (CoT) and PromptTemplate (legacy)
                if isinstance(prompt_template, ChatPromptTemplate):
                    # For ChatPromptTemplate, we'll format it directly per question
                    template_str = None
                elif hasattr(prompt_template, 'template'):
                    template_str = prompt_template.template
                else:
                    template_str = str(prompt_template)
                
                # Process each question
                progress_bar = st.progress(0)
                for idx, qdata in enumerate(questions):
                    question_text = qdata["question_text"]
                    reference_answer = qdata["answer_text"]
                    
                    try:
                        # Generate RAG answer with smart retrieval
                        source_docs = smart_rag_retrieval(
                            retriever=st.session_state.rag_retriever,
                            query=question_text,
                            target_k=target_k,
                            expansion_factor=1.8  # Retrieve 80% more documents initially
                        )
                        
                        # Create context and generate answer
                        context = "\n\n".join([doc.page_content for doc in source_docs])
                        
                        # Format prompt based on template type
                        if isinstance(prompt_template, ChatPromptTemplate):
                            formatted_prompt = prompt_template.format(question=question_text, context=context)
                        else:
                            formatted_prompt = template_str.format(question=question_text, context=context)
                        
                        answer = llm.invoke(formatted_prompt).content
                        
                        # Evaluate answer
                        import asyncio
                        evaluation = asyncio.run(
                            st.session_state.evaluator.evaluate_single_response(answer, reference_answer)
                        )
                        
                        # Create result object
                        result = {
                            "question_id": qdata.get("id"),
                            "year": qdata.get("year"),
                            "question_number": qdata.get("question_number"),
                            "question": question_text,
                            "generated_answer": answer,
                            "reference_answer": reference_answer,
                            "evaluation": evaluation,
                            "llm_provider": selected_llm_provider,
                            "llm_model": selected_llm_model,
                            "response_type": response_type,
                            "rag": True,
                            "timestamp": datetime.now().isoformat(),
                            "retrieved_context": context,
                            "source_docs": [doc.metadata for doc in source_docs],
                            "source_file": qdata.get("source_file", ""),
                            "rag_config": st.session_state.get("rag_config", {}),
                        }
                        results.append(result)
                        
                    except Exception as e:
                        st.error(f"Error processing question {idx + 1}: {str(e)}")
                        continue
                    
                    progress_bar.progress((idx + 1) / len(questions))
                
                progress_bar.empty()
                
                # Run additional evaluations if enabled
                if st.session_state.get("enable_rag_eval", False) and results:
                    results = run_langchain_rag_evaluation(results)
                
                if st.session_state.get("enable_ragas_eval", False) and results:
                    results = run_ragas_rag_evaluation(results)
                
                if results:
                    # Store results in both histories for backward compatibility and export functionality
                    st.session_state.rag_evaluation_history.extend(results)
                    st.session_state.evaluation_history.extend(results)
                    
                    # Calculate aggregate scores
                    aggregate_scores = st.session_state.evaluator.calculate_aggregate_scores(results)
                    
                    st.success(f"✅ Batch RAG evaluation completed! Processed {len(results)} questions")
                    
                    # Display aggregate metrics
                    st.subheader("📊 Results Summary")
                    col_a, col_b, col_c, col_d, col_e = st.columns(5)

                    with col_a:
                        st.metric("Total Questions", len(results))
                    with col_b:
                        st.metric("Avg BLEU", f"{aggregate_scores['Avg BLEU']:.3f}")
                    with col_c:
                        st.metric("Avg ROUGE", f"{aggregate_scores['Avg ROUGE']:.3f}")
                    with col_d:
                        st.metric("Avg String Similarity", f"{aggregate_scores['Avg String Similarity']:.3f}")
                    with col_e:
                        if 'Avg Semantic Similarity' in aggregate_scores:
                            st.metric("Avg Semantic Similarity", f"{aggregate_scores['Avg Semantic Similarity']:.3f}")
                    
                    # Results table preview
                    st.subheader("📋 Results Preview")
                    display_results_preview(results)
                    
                    # Export note - immediate export available below
                    st.info("💡 **Export Results:** Export buttons are available below and in the sidebar. All results from this session are automatically accumulated.")
                    
                    # Detailed analysis in expander
                    with st.expander("📊 Detailed Analysis", expanded=False):
                        display_detailed_analysis(results, selected_llm_provider, selected_llm_model, response_type)
                    
                    # Add immediate export options right after batch results
                    st.markdown("---")
                    st.subheader("📥 Export This Session")
                    st.caption(f"Export all {len(st.session_state.evaluation_history)} accumulated results")
                    
                    col_batch_exp1, col_batch_exp2 = st.columns(2)
                    with col_batch_exp1:
                        if st.button("📥 Download Excel Report", key="batch_excel_export"):
                            export_results(st.session_state.evaluation_history, "excel")
                    with col_batch_exp2:
                        if st.button("📥 Download JSON Report", key="batch_json_export"):
                            export_results(st.session_state.evaluation_history, "json")
                
                else:
                    st.error("❌ Batch RAG evaluation failed - no results generated")
    
    with col2:
        display_rag_batch_sidebar_info(selected_llm_provider, selected_llm_model)



def display_results_preview(results):
    """Display a preview table of results"""
    preview_data = []
    for i, result in enumerate(results[:10]):  # Show first 10
        row = {
            "Question #": i + 1,
            "Year": result.get("year", "N/A"),
            "Q#": result.get("question_number", "N/A"),
            "BLEU": f"{result['evaluation']['bleu_score']:.3f}",
            "ROUGE": f"{result['evaluation']['rouge_score']:.3f}",
            "Str Sim": f"{result['evaluation']['string_similarity_score']:.3f}",
            "Sem Sim": f"{result['evaluation'].get('semantic_similarity_score', float('nan')):.3f}",
            "Question Preview": (result["question"][:50] + "..." if len(result["question"]) > 50 else result["question"])
        }
        
        # Add LangChain scores if available
        if "langchain_evaluation" in result:
            lc_eval = result["langchain_evaluation"]
            row.update({
                "LC Correctness": f"{lc_eval.get('correctness', {}).get('score', 'N/A')}/5",
                "LC Relevance": f"{lc_eval.get('relevance', {}).get('score', 'N/A')}/5",
                "LC Grounded": f"{lc_eval.get('groundedness', {}).get('score', 'N/A')}/5",
                "LC Retrieval": f"{lc_eval.get('retrieval_relevance', {}).get('score', 'N/A')}/5",
            })
        
        # Add RAGAS scores if available
        if "ragas_evaluation" in result:
            ragas_eval = result["ragas_evaluation"]
            row.update({
                "RG Recall": f"{ragas_eval.get('context_recall', {}).get('score', 'N/A'):.3f}",
                "RG Precision": f"{ragas_eval.get('context_precision', {}).get('score', 'N/A'):.3f}",
                "RG Faithful": f"{ragas_eval.get('faithfulness', {}).get('score', 'N/A'):.3f}",
                "RG Factual": f"{ragas_eval.get('factual_correctness', {}).get('score', 'N/A'):.3f}",
                "RG Accuracy": f"{ragas_eval.get('answer_accuracy', {}).get('score', 'N/A'):.3f}",
            })
        
        preview_data.append(row)
    
    if preview_data:
        df_preview = pd.DataFrame(preview_data)
        st.dataframe(df_preview, use_container_width=True)
        
        if len(results) > 10:
            st.caption(f"Showing first 10 of {len(results)} results. Download full results using export buttons above.")


def display_detailed_analysis(results, llm_provider, llm_model, response_type):
    """Display detailed analysis of results"""
    
    # Score distribution analysis
    st.subheader("Score Distribution")
    
    scores_data = {
    "BLEU": [r["evaluation"]["bleu_score"] for r in results],
    "ROUGE": [r["evaluation"]["rouge_score"] for r in results],
    "String Sim": [r["evaluation"]["string_similarity_score"] for r in results],
    }
    if any('semantic_similarity_score' in r['evaluation'] for r in results):
        scores_data["Semantic Sim"] = [r["evaluation"].get("semantic_similarity_score", 0.0) for r in results]
    
    for metric, scores in scores_data.items():
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.write(f"**{metric} Score Statistics:**")
            st.write(f"Mean: {np.mean(scores):.3f}")
            st.write(f"Median: {np.median(scores):.3f}")
            st.write(f"Std Dev: {np.std(scores):.3f}")
        with col_dist2:
            st.write(f"Min: {np.min(scores):.3f}, Max: {np.max(scores):.3f}")
            
            # Score distribution
            high_scores = sum(1 for s in scores if s > 0.5)
            medium_scores = sum(1 for s in scores if 0.2 <= s <= 0.5)
            low_scores = sum(1 for s in scores if s < 0.2)
            
            st.write(f"High (>0.5): {high_scores}")
            st.write(f"Medium (0.2-0.5): {medium_scores}")
            st.write(f"Low (<0.2): {low_scores}")
    
    # Configuration summary
    st.subheader("⚙️ Configuration")
    st.write(f"**LLM Used:** {llm_provider}/{llm_model}")
    st.write(f"**Response Type:** {response_type}")
    st.write(f"**Questions Processed:** {len(results)}")
    
    # RAG Configuration
    if results and results[0].get("rag_config"):
        rag_config = results[0]["rag_config"]
        st.write("**RAG Configuration:**")
        st.write(f"• Embedding Model: {rag_config.get('embedding_provider', 'N/A')}/{rag_config.get('embedding_model', 'N/A')}")
        st.write(f"• Retrieval K: {rag_config.get('k_value', 'N/A')}")
    
    st.write(f"**Evaluation Method:** Direct database matching (no embeddings)")

# Improved export function
def run_langchain_rag_evaluation(results: List[Dict]) -> List[Dict]:
    """
    Run LangChain RAG evaluation on results if enabled
    
    Args:
        results: List of RAG evaluation results
        
    Returns:
        List of results with LangChain evaluation added
    """
    if not st.session_state.get("enable_rag_eval", False):
        return results
    
    # Get evaluation model from session state widgets
    eval_provider = st.session_state.get("eval_llm_provider")
    eval_model = st.session_state.get("eval_llm_model")
    
    if not eval_provider or not eval_model:
        st.warning("LangChain RAG evaluation enabled but no evaluation model selected")
        return results
    
    try:
        # Create evaluation LLM
        eval_llm = ModelManager.create_llm(eval_provider, eval_model)
        if not eval_llm:
            st.error(f"❌ Failed to create LangChain evaluation LLM ({eval_provider}/{eval_model})")
            st.warning("💡 **LangChain Evaluation Skipped:** The evaluation LLM could not be created. Check the model availability.")
            return results
        
        # Initialize evaluation pipeline
        eval_pipeline = LangchainEvalPipeline(eval_llm)
        
        # Filter only RAG results - check for retrieved_context or source_docs as indicators
        rag_results = []
        non_rag_results = []
        
        for r in results:
            # Check if this is a RAG result by looking for RAG-specific fields
            if (r.get("rag", False) or 
                r.get("retrieved_context") or 
                r.get("source_docs") or
                "retrieved_context" in r or
                "source_docs" in r):
                rag_results.append(r)
            else:
                non_rag_results.append(r)
        
        if not rag_results:
            st.info("No RAG results found for LangChain evaluation")
            return results
        
        st.info(f"Running LangChain RAG evaluation on {len(rag_results)} RAG results...")
        
        # Prepare samples for batch evaluation
        samples = []
        for result in rag_results:
            samples.append({
                "question": result.get("question", ""),
                "reference_answer": result.get("reference_answer", ""),
                "generated_answer": result.get("generated_answer", ""),
                "context": result.get("retrieved_context", "")
            })
        
        # Run batch evaluation
        with st.spinner("Running LangChain RAG evaluation..."):
            eval_results = eval_pipeline.batch_evaluate(samples)
        
        # Merge evaluation results back into original results
        enhanced_rag_results = []
        for i, result in enumerate(rag_results):
            enhanced_result = result.copy()
            if i < len(eval_results):
                enhanced_result["langchain_evaluation"] = eval_results[i]["evaluation"]
            enhanced_rag_results.append(enhanced_result)
        
        # Combine with non-RAG results
        final_results = enhanced_rag_results + non_rag_results
        
        st.success(f"✅ LangChain evaluation completed for {len(rag_results)} RAG results")
        return final_results
        
    except Exception as e:
        st.error(f"Error during LangChain evaluation: {str(e)}")
        return results

def run_ragas_rag_evaluation(results: List[Dict]) -> List[Dict]:
    """
    Run RAGAS RAG evaluation on results if enabled
    
    Args:
        results: List of RAG evaluation results
        
    Returns:
        List of results with RAGAS evaluation added
    """
    if not st.session_state.get("enable_ragas_eval", False):
        return results
    
    # Get evaluation model from session state widgets
    ragas_provider = st.session_state.get("ragas_llm_provider")
    ragas_model = st.session_state.get("ragas_llm_model")
    
    if not ragas_provider or not ragas_model:
        st.warning("RAGAS RAG evaluation enabled but no evaluation model selected")
        return results
    
    try:
        # Create evaluation LLM
        eval_llm = ModelManager.create_llm(ragas_provider, ragas_model)
        if not eval_llm:
            st.error(f"❌ Failed to create RAGAS evaluation LLM ({ragas_provider}/{ragas_model})")
            st.warning("💡 **RAGAS Evaluation Skipped:** The evaluation LLM could not be created. Check the model availability.")
            return results
        
        # Create embeddings model if enabled
        eval_embeddings = None
        if st.session_state.get("ragas_use_embeddings", False):
            ragas_emb_provider = st.session_state.get("ragas_emb_provider")
            ragas_emb_model = st.session_state.get("ragas_emb_model")
            
            if ragas_emb_provider and ragas_emb_model:
                eval_embeddings = EmbeddingManager.create_embedding_model(ragas_emb_provider, ragas_emb_model)
                if not eval_embeddings:
                    st.warning("Failed to create RAGAS embeddings model - response relevancy will be disabled")
        
        # Initialize RAGAS evaluation pipeline
        eval_pipeline = RAGASEvalPipeline(eval_llm, eval_embeddings)
        
        # Filter only RAG results - check for retrieved_context or source_docs as indicators
        rag_results = []
        non_rag_results = []
        
        for r in results:
            # Check if this is a RAG result by looking for RAG-specific fields
            if (r.get("rag", False) or 
                r.get("retrieved_context") or 
                r.get("source_docs") or
                "retrieved_context" in r or
                "source_docs" in r):
                rag_results.append(r)
            else:
                non_rag_results.append(r)
        
        if not rag_results:
            st.info("No RAG results found for RAGAS evaluation")
            return results
        
        st.info(f"Running RAGAS RAG evaluation on {len(rag_results)} RAG results...")
        
        # Prepare samples for batch evaluation
        samples = []
        for result in rag_results:
            samples.append({
                "question": result.get("question", ""),
                "reference_answer": result.get("reference_answer", ""),
                "generated_answer": result.get("generated_answer", ""),
                "context": result.get("retrieved_context", "")
            })
        
        # Run batch evaluation using async approach
        with st.spinner("Running RAGAS RAG evaluation..."):
            eval_results = asyncio.run(eval_pipeline.batch_evaluate(samples))
        
        # Merge evaluation results back into original results
        enhanced_rag_results = []
        for i, result in enumerate(rag_results):
            enhanced_result = result.copy()
            if i < len(eval_results):
                enhanced_result["ragas_evaluation"] = eval_results[i]["evaluation"]
            enhanced_rag_results.append(enhanced_result)
        
        # Combine with non-RAG results
        final_results = enhanced_rag_results + non_rag_results
        
        st.success(f"✅ RAGAS evaluation completed for {len(rag_results)} RAG results")
        return final_results
        
    except Exception as e:
        st.error(f"Error during RAGAS evaluation: {str(e)}")
        return results

def export_results(results: List[Dict], format_type: str = "excel"):
    """Export results to Excel or JSON with proper error handling"""
    if not results:
        st.warning("❌ No results to export - results list is empty")
        return
    
    # Prepare export
    rag_count = sum(1 for r in results if r.get("rag", False))
    ragas_count = sum(1 for r in results if "ragas_evaluation" in r)
    langchain_count = sum(1 for r in results if "langchain_evaluation" in r)
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "excel":
            # Prepare data for Excel export
            df_data = []
            for result in results:
                row = {
                    "timestamp": result.get("timestamp", ""),
                    "question_id": result.get("question_id", ""),
                    "year": result.get("year", ""),
                    "question_number": result.get("question_number", ""),
                    "llm_provider": result.get("llm_provider", ""),
                    "llm_model": result.get("llm_model", ""),
                    "response_type": result.get("response_type", ""),
                    "question": result.get("question", ""),
                    "generated_answer": result.get("generated_answer", ""),
                    "reference_answer": result.get("reference_answer", ""),
                    "bleu_score": result.get("evaluation", {}).get("bleu_score", 0),
                    "rouge_score": result.get("evaluation", {}).get("rouge_score", 0),
                    "string_similarity_score": result.get("evaluation", {}).get("string_similarity_score", 0),
                    "semantic_similarity_score": result.get("evaluation", {}).get("semantic_similarity_score", 0),
                    "source_file": result.get("source_file", ""),
                    # RAG configuration fields
                    "rag_embedding_provider": result.get("rag_config", {}).get("embedding_provider", ""),
                    "rag_embedding_model": result.get("rag_config", {}).get("embedding_model", ""),
                    "rag_retrieval_method": result.get("rag_config", {}).get("retrieval_method", ""),
                    "rag_is_hybrid": result.get("rag_config", {}).get("is_hybrid", ""),
                    "rag_bm25_weight": result.get("rag_config", {}).get("bm25_weight", ""),
                    "rag_vector_weight": result.get("rag_config", {}).get("vector_weight", ""),
                    "rag_use_reranking": result.get("rag_config", {}).get("use_reranking", ""),
                    "rag_rerank_type": result.get("rag_config", {}).get("rerank_type", ""),
                    "rag_rerank_model": result.get("rag_config", {}).get("rerank_model", ""),
                    "rag_k_value": result.get("rag_config", {}).get("k_value", "")
                }
                
                # Add LangChain evaluation metrics if available
                if "langchain_evaluation" in result:
                    lc_eval = result["langchain_evaluation"]
                    row.update({
                        "lc_correctness_score": lc_eval.get("correctness", {}).get("score", ""),
                        # "lc_correctness_explanation": lc_eval.get("correctness", {}).get("explanation", ""),  # Commented out for zero-shot Q&A
                        "lc_relevance_score": lc_eval.get("relevance", {}).get("score", ""),
                        # "lc_relevance_explanation": lc_eval.get("relevance", {}).get("explanation", ""),  # Commented out for zero-shot Q&A
                        "lc_groundedness_score": lc_eval.get("groundedness", {}).get("score", ""),
                        # "lc_groundedness_explanation": lc_eval.get("groundedness", {}).get("explanation", ""),  # Commented out for zero-shot Q&A
                        "lc_retrieval_relevance_score": lc_eval.get("retrieval_relevance", {}).get("score", ""),
                        # "lc_retrieval_relevance_explanation": lc_eval.get("retrieval_relevance", {}).get("explanation", ""),  # Commented out for zero-shot Q&A
                    })
                
                # Add RAGAS evaluation metrics if available
                if "ragas_evaluation" in result:
                    ragas_eval = result["ragas_evaluation"]
                    row.update({
                        "ragas_context_recall_score": ragas_eval.get("context_recall", {}).get("score", ""),
                        "ragas_context_precision_score": ragas_eval.get("context_precision", {}).get("score", ""),
                        # "ragas_context_entity_recall_score": ragas_eval.get("context_entity_recall", {}).get("score", ""),
                        "ragas_faithfulness_score": ragas_eval.get("faithfulness", {}).get("score", ""),
                        "ragas_factual_correctness_score": ragas_eval.get("factual_correctness", {}).get("score", ""),
                        "ragas_answer_accuracy_score": ragas_eval.get("answer_accuracy", {}).get("score", ""),
                    })
                
                df_data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(df_data)
            filename = f"llm_evaluation_results_{timestamp}.xlsx"
            
            # Create Excel file in memory
            with st.spinner("Creating Excel file..."):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Evaluation_Results')
                
                # Reset buffer position to beginning
                output.seek(0)
                
                # Get the Excel data
                excel_data = output.getvalue()
                
                # Verify we have data
                if len(excel_data) == 0:
                    st.error("❌ Excel file is empty - something went wrong during creation")
                    return
            
            # Provide download button
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success(f"✅ Excel report ready for download: {filename}")
        
        else:  # JSON format
            filename = f"llm_evaluation_results_{timestamp}.json"
            json_data = json.dumps(results, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Download JSON Report",
                data=json_data,
                file_name=filename,
                mime="application/json"
            )
            
            st.success(f"✅ JSON report ready for download: {filename}")
    
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")
        st.error(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        
        # Additional debugging info
        st.error(f"📊 Debug info:")
        st.error(f"  - Results length: {len(results) if results else 'N/A'}")
        st.error(f"  - Format type: {format_type}")
        st.error(f"  - Timestamp: {timestamp if 'timestamp' in locals() else 'N/A'}")
        
        # Show the actual error traceback for debugging
        import traceback
        st.error("🔍 Full traceback:")
        st.code(traceback.format_exc())


def _setup_rag(file_obj, persist_dir, emb_provider, emb_model, k_value, is_hybrid, use_reranking, rerank_type, rerank_model):
    # Save uploaded file to temporary path
    tmp_path = Path(tempfile.gettempdir()) / f"tmp_{file_obj.name}"
    tmp_path.write_bytes(file_obj.getbuffer())

    # Create embedding model
    embed_obj = EmbeddingManager.create_embedding_model(emb_provider, emb_model)
    if embed_obj is None:
        st.error("Failed to create RAG embedding model.")
        return

    col_map = {
        "citing": {
            "text" : "TEXT_FROM",
            "celex": "CELEX_FROM",
            "para" : "NUMBER_FROM",
            "title": "TITLE_FROM",
            "date": "DATE_FROM",
        },
        "cited": {
            "text" : "TEXT_TO",
            "celex": "CELEX_TO",
            "para" : "NUMBER_TO",
            "title": "TITLE_TO",
            "date": "DATE_TO",
        },
    }
    
    # Check dataset size and choose processing method
    try:
        # Quick size check by reading just the header
        import pandas as pd
        if tmp_path.suffix.lower() == '.csv':
            # Read just first few rows to estimate size
            sample_df = pd.read_csv(tmp_path, nrows=10)
            total_lines = sum(1 for line in open(tmp_path, 'r', encoding='utf-8', errors='ignore'))
            estimated_rows = total_lines - 1  # Subtract header
        else:
            # For JSON files, try to estimate
            estimated_rows = sum(1 for line in open(tmp_path, 'r', encoding='utf-8', errors='ignore'))
        
        # Decision threshold: use streaming for datasets > 50K rows
        use_streaming = estimated_rows > 50000
        
        if use_streaming:
            # Import the appropriate streaming pipeline
            if is_hybrid:
                # Use hybrid streaming pipeline
                create_pipeline_func = create_large_hybrid_dataset_pipeline
                pipeline_type = "hybrid streaming"
            else:
                # Use regular streaming pipeline
                from RAG_Pipeline.rag_pipeline import create_large_dataset_pipeline
                create_pipeline_func = create_large_dataset_pipeline
                pipeline_type = "dense streaming"
            
            # Create progress containers
            progress_container = st.empty()
            status_container = st.empty()
            
            # Custom progress callback for UI updates
            import logging
            
            class StreamlitProgressHandler(logging.Handler):
                def __init__(self, progress_container, status_container):
                    super().__init__()
                    self.progress_container = progress_container
                    self.status_container = status_container
                    self.total_batches = None
                    self.current_batch = 0
                    
                def emit(self, record):
                    msg = record.getMessage()
                    
                    # Extract batch information from log messages
                    if "Processing batch" in msg:
                        try:
                            # Extract current batch number
                            if "/" in msg:
                                batch_info = msg.split("Processing batch ")[1].split(" ")[0]
                                if "/" in batch_info:
                                    current, total = batch_info.split("/")
                                    self.current_batch = int(current)
                                    if self.total_batches is None:
                                        self.total_batches = int(total)
                                    
                                    # Update progress bar
                                    progress = self.current_batch / self.total_batches
                                    self.progress_container.progress(
                                        progress, 
                                        text=f"Processing batch {self.current_batch}/{self.total_batches}"
                                    )
                        except:
                            pass
                    
                    elif "Processing chunk" in msg:
                        self.status_container.info(f"🔄 {msg}")
                    elif "Successfully added batch" in msg:
                        self.status_container.success(f"✅ {msg}")
                    elif "Error processing batch" in msg:
                        self.status_container.error(f"❌ {msg}")
                    
                    # Handle hybrid pipeline specific messages
                    elif "Initializing large hybrid dataset pipeline" in msg:
                        self.status_container.info(f"🔄 {msg}")
                    elif "BM25 indexing progress" in msg:
                        self.status_container.info(f"🔄 {msg}")
                    elif "BM25 retriever created successfully" in msg:
                        self.status_container.success(f"✅ {msg}")
                    elif "Ensemble retriever created successfully" in msg:
                        self.status_container.success(f"✅ {msg}")
                    elif "Reranking enabled successfully" in msg:
                        self.status_container.success(f"✅ {msg}")
                    elif "CRITICAL" in msg or "ERROR" in msg.upper():
                        self.status_container.error(f"❌ {msg}")
                    elif "WARNING" in msg.upper():
                        self.status_container.warning(f"⚠️ {msg}")
                    elif "SUCCESS" in msg:
                        self.status_container.success(f"✅ {msg}")
                    elif record.levelname in ["INFO"] and any(keyword in msg for keyword in ["retriever", "pipeline", "initialized"]):
                        self.status_container.info(f"ℹ️ {msg}")
            
            # Set up progress logging
            logger = logging.getLogger("RAG_Pipeline")
            progress_handler = StreamlitProgressHandler(progress_container, status_container)
            progress_handler.setLevel(logging.INFO)
            logger.addHandler(progress_handler)
            logger.setLevel(logging.INFO)
            
            # Also set up logging for the hybrid variant module
            hybrid_logger = logging.getLogger("RAG_Pipeline.variants.hybrid")
            hybrid_logger.addHandler(progress_handler)
            hybrid_logger.setLevel(logging.INFO)
            
            try:
                # Create streaming pipeline
                pipeline_name = "hybrid vector store" if is_hybrid else "vector store"
                with st.spinner(f"🏗️ Building {pipeline_name} with batch processing..."):
                    base_params = {
                        "dataset_path": tmp_path,
                        "persist_dir": Path(persist_dir),
                        "embedding": embed_obj,
                        "batch_size": 2000,  # Maximum batch size for premium rate limits
                        "chunk_size": 10000,  # Large chunks for maximum efficiency
                        "force_rebuild": False,
                        "col_map": col_map,
                        "doc_format_style": "legal_standard",  # Always use legal_standard
                        "include_celex_in_content": True,  # Always include CELEX
                        "k": k_value,
                    }
                    
                    # Add hybrid-specific parameters if needed
                    if is_hybrid:
                        base_params.update({
                            "bm25_weight": 0.3,
                            "vector_weight": 0.7,
                            "use_reranking": use_reranking,
                            "rerank_type": rerank_type,
                            "rerank_model": rerank_model,
                        })
                    
                    rag = create_pipeline_func(**base_params)
                
                # Clean up progress display
                progress_container.empty()
                status_container.empty()
                
                # Remove the progress handler
                logger.removeHandler(progress_handler)
                hybrid_logger.removeHandler(progress_handler)
                
                # Verify retriever was created properly
                if hasattr(rag, 'retriever') and rag.retriever is not None:
                    st.success(f"✅ Retriever created successfully")
                else:
                    st.error("❌ Pipeline initialization failed")
                    return
                
            except Exception as e:
                # Clean up on error
                progress_container.empty()
                status_container.empty()
                logger.removeHandler(progress_handler)
                try:
                    hybrid_logger.removeHandler(progress_handler)
                except:
                    pass  # Ignore if handler wasn't added
                st.error(f"❌ Error during batch processing: {str(e)}")
                st.error(f"🔍 Error details: {type(e).__name__}: {str(e)}")
                return
        
        else:
            pipeline_type = "hybrid" if is_hybrid else "dense"
            
            # Use appropriate pipeline for smaller datasets
            base_params = {
                "dataset_path": tmp_path,
                "persist_dir": Path(persist_dir),
                "embedding": embed_obj,
                "k": k_value,
                "force_rebuild": False,
                "col_map": col_map,
                "doc_format_style": "legal_standard",  # Always use legal_standard
                "include_celex_in_content": True,  # Always include CELEX
            }
            
            if is_hybrid:
                # Create hybrid pipeline
                base_params.update({
                    "bm25_weight": 0.3,
                    "vector_weight": 0.7,
                    "use_reranking": use_reranking,
                    "rerank_type": rerank_type,
                    "rerank_model": rerank_model,
                })
                rag = HybridRAGPipeline(**base_params)
            else:
                # Create regular dense pipeline
                rag = RAGPipeline(**base_params)
            
            pipeline_name = "hybrid vector store" if is_hybrid else "vector store"
            with st.spinner(f"Building / loading {pipeline_name}…"):
                rag.initialise()
                
            # Verify retriever was created properly
            if not hasattr(rag, 'retriever') or rag.retriever is None:
                st.error("❌ Pipeline initialization failed")
                return
    
    except Exception as e:
        st.error(f"❌ Error processing dataset: {str(e)}")
        # Fallback to regular processing
        fallback_type = "hybrid" if is_hybrid else "dense"
        st.warning(f"🔄 Falling back to regular {fallback_type} processing...")
        
        base_params = {
            "dataset_path": tmp_path,
            "persist_dir": Path(persist_dir),
            "embedding": embed_obj,
            "k": k_value,
            "force_rebuild": False,
            "col_map": col_map,
            "doc_format_style": "legal_standard",  # Always use legal_standard
            "include_celex_in_content": True,  # Always include CELEX
        }
        
        if is_hybrid:
            # Create hybrid pipeline
            base_params.update({
                "bm25_weight": 0.3,
                "vector_weight": 0.7,
                "use_reranking": use_reranking,
                "rerank_type": rerank_type,
                "rerank_model": rerank_model,
            })
            rag = HybridRAGPipeline(**base_params)
        else:
            # Create regular dense pipeline
            rag = RAGPipeline(**base_params)
        
        pipeline_name = "hybrid vector store" if is_hybrid else "vector store"
        with st.spinner(f"Building / loading {pipeline_name}…"):
            rag.initialise()


    # Store RAG system references
    st.session_state.rag_system    = rag
    st.session_state.rag_retriever = rag.retriever
    st.session_state.rag_title_idx = rag.title_index
    

    
    # Store RAG configuration in session state for reference
    st.session_state.rag_config = {
        "embedding_provider": emb_provider,
        "embedding_model": emb_model,
        "doc_format_style": "legal_standard",  # Always legal_standard
        "include_celex_in_content": True,  # Always include CELEX
        "k_value": k_value,
        "retrieval_method": "hybrid" if is_hybrid else "dense",
        "is_hybrid": is_hybrid,
        "use_reranking": use_reranking,
        "rerank_type": rerank_type if use_reranking else None,
        "rerank_model": rerank_model if use_reranking else None,
        "bm25_weight": 0.3 if is_hybrid else None,
        "vector_weight": 0.7 if is_hybrid else None,
        "dataset_size": estimated_rows if 'estimated_rows' in locals() else "unknown",
        "processing_method": "streaming_batch" if use_streaming else "standard"
    }
    
    # Display completion message with statistics
    try:
        stats = rag.get_stats()
        
        # Determine actual retrieval method based on stats
        if isinstance(stats, dict):
            actual_retrieval_type = stats.get("retrieval_type", "unknown")
            if actual_retrieval_type == "hybrid":
                bm25_docs = stats.get("bm25_docs_indexed", 0)
                if bm25_docs and bm25_docs != "unknown" and bm25_docs > 0:
                    retrieval_info = f"Hybrid (0.3 BM25 + 0.7 Vector)"
                    bm25_status = f"✅ BM25 index: {bm25_docs:,} documents"
                else:
                    retrieval_info = f"Dense (Vector Only) - BM25 fallback"
                    bm25_status = "⚠️ BM25 indexing failed, using vector-only"
            else:
                retrieval_info = f"Dense (Vector Only)"
                bm25_status = None
        else:
            # Fallback to original logic
            retrieval_info = f"{'Hybrid (0.3 BM25 + 0.7 Vector)' if is_hybrid else 'Dense (Vector Only)'}"
            bm25_status = None
        
        rerank_info = f" + {rerank_type.upper()} Reranking" if use_reranking else ""
        
        if isinstance(stats, dict) and "total_documents" in stats:
            st.success(f"✅ RAG ready! Method: {retrieval_info}{rerank_info}, K={k_value}")
            st.info(f"📊 Vector store contains {stats['total_documents']:,} documents")
            
            # Show BM25 status if available
            if bm25_status:
                if "failed" in bm25_status:
                    st.warning(bm25_status)
                else:
                    st.info(bm25_status)
            
            # Show dataset processing method
            processing_method = stats.get("processing_method", "standard")
            dataset_size = stats.get("dataset_size", "unknown")
            if processing_method == "streaming_batch" and dataset_size != "unknown":
                st.info(f"🚀 Processed {dataset_size:,} rows using streaming batch processing")
        else:
            st.success(f"✅ RAG ready! Method: {retrieval_info}{rerank_info}, K={k_value}")
    except Exception as stats_error:
        retrieval_info = f"{'Hybrid' if is_hybrid else 'Dense'}"
        rerank_info = f" + {rerank_type.upper()}" if use_reranking else ""
        st.success(f"✅ RAG ready! Method: {retrieval_info}{rerank_info}, K={k_value}")
        st.warning(f"Could not retrieve detailed stats: {stats_error}")

def _reset_rag(persist_dir):
    """Reset the RAG system by clearing the directory."""
    try:
        # Close any open ChromaDB connections
        if st.session_state.get("rag_system"):
            try:
                if hasattr(st.session_state.rag_system, 'retriever') and st.session_state.rag_system.retriever:
                    vectorstore = st.session_state.rag_system.retriever.vectorstore
                    if hasattr(vectorstore, '_client'):
                        vectorstore._client = None
                    if hasattr(vectorstore, '_collection'):
                        vectorstore._collection = None
            except Exception as e:
                st.warning(f"Warning during cleanup: {e}")
        
        # Clear session state references
        st.session_state.rag_system = None
        st.session_state.rag_retriever = None
        st.session_state.rag_title_idx = None
        if "rag_config" in st.session_state:
            del st.session_state.rag_config
        
        # Force garbage collection and allow file handle release
        import gc
        import time
        gc.collect()
        time.sleep(0.5)
        
        # Try to remove the directory
        if os.path.exists(persist_dir):
            try:
                shutil.rmtree(persist_dir)
                st.success(f"✅ RAG system reset! Removed {persist_dir}")
            except PermissionError as e:
                st.error(f"❌ Permission error: {str(e)}")
                st.warning("💡 **Solution**: Restart the Streamlit app to fully release file handles, then try reset again.")
                st.info("This is a Windows limitation where ChromaDB files remain locked by the process.")
            except Exception as e:
                st.error(f"❌ Error removing directory: {str(e)}")
        else:
            st.info("No RAG data to reset.")
            
    except Exception as e:
        # Clear session state even if cleanup fails
        st.session_state.rag_system = None
        st.session_state.rag_retriever = None
        st.session_state.rag_title_idx = None
        if "rag_config" in st.session_state:
            del st.session_state.rag_config
        st.error(f"❌ Error during reset: {str(e)}")
        st.warning("💡 Session state cleared. Restart the app if file deletion issues persist.")

def _soft_reset_rag():
    """Soft reset - clear session state only, keep database files."""
    try:
        # Clear session state references
        st.session_state.rag_system = None
        st.session_state.rag_retriever = None
        st.session_state.rag_title_idx = None
        if "rag_config" in st.session_state:
            del st.session_state.rag_config
        
        # Force garbage collection
        import gc
        gc.collect()
        
        st.success("✅ RAG session state cleared! Database files preserved.")
        st.info("💡 Use 'Build / Load' to reconnect to existing database or upload new data.")
        
    except Exception as e:
        st.error(f"❌ Error during soft reset: {str(e)}")

def display_rag_batch_sidebar_info(selected_llm_provider, selected_llm_model):
    """Display sidebar info for batch RAG Q&A evaluation (no Tips section)"""
    st.subheader("⚙️ Batch Settings (RAG Q&A)")
    
    # RAG System Status
    if st.session_state.rag_system and st.session_state.get("rag_config"):
        rag_config = st.session_state.rag_config
        st.success("🟢 RAG System Ready")
        st.write("**RAG Configuration:**")
        
        # Retrieval method info
        retrieval_method = rag_config.get('retrieval_method', 'dense')
        st.write(f"• Retrieval: {retrieval_method.title()}")
        
        if rag_config.get('is_hybrid', False):
            bm25_weight = rag_config.get('bm25_weight', 0.3)
            vector_weight = rag_config.get('vector_weight', 0.7)
            st.write(f"• Weights: {vector_weight} Vector + {bm25_weight} BM25")
            
            if rag_config.get('use_reranking', False):
                rerank_type = rag_config.get('rerank_type', 'unknown')
                rerank_model = rag_config.get('rerank_model', 'unknown')
                st.write(f"• Reranker: {rerank_type.upper()} ({rerank_model})")
            else:
                st.write("• Reranker: None")
        
        st.write(f"• K-value: {rag_config.get('k_value', 'N/A')}")
        st.write(f"• Embedding: {rag_config.get('embedding_provider', 'N/A')}/{rag_config.get('embedding_model', 'N/A')}")
        
        # Show processing method and dataset info
        dataset_size = rag_config.get('dataset_size', 'unknown')
        processing_method = rag_config.get('processing_method', 'standard')
        
        if dataset_size != 'unknown':
            st.write(f"• Dataset size: {dataset_size:,} rows")
        
        if processing_method == 'streaming_batch':
            st.info("⚡ **Batch Processing Used** - Large dataset processed efficiently")
        else:
            st.info("📊 **Standard Processing** - Regular dataset processing")
            
    elif st.session_state.rag_system:
        st.info("🟡 RAG System Loaded (Legacy Config)")
    else:
        st.warning("🔴 RAG System Not Ready")
    
    # Statistics about available questions
    if st.session_state.question_bank_ready:
        total_questions = len(st.session_state.evaluator.question_bank.questions)
        st.metric("Available Questions", total_questions)
        st.info("🟢 Database Ready")
    # Current session results
    if st.session_state.rag_evaluation_history:
        st.markdown("---")
        st.subheader("📊 Session Results (RAG Q&A)")
        # Filter current model results
        current_results = [r for r in st.session_state.rag_evaluation_history 
                          if r.get("llm_provider") == selected_llm_provider 
                          and r.get("llm_model") == selected_llm_model]
        if current_results:
            st.write(f"**Current Model:** {len(current_results)} evaluations")
            
            # Traditional metrics
            avg_scores = {
                'bleu': np.mean([r['evaluation']['bleu_score'] for r in current_results]),
                'rouge': np.mean([r['evaluation']['rouge_score'] for r in current_results]),
                'string': np.mean([r['evaluation']['string_similarity_score'] for r in current_results]),
            }
            if any('semantic_similarity_score' in r['evaluation'] for r in current_results):
                avg_scores['semantic'] = np.mean([r['evaluation'].get('semantic_similarity_score', 0.0) for r in current_results])
            
            st.write("**Traditional Metrics:**")
            st.write(f"Avg BLEU: {avg_scores['bleu']:.3f}")
            st.write(f"Avg ROUGE: {avg_scores['rouge']:.3f}")
            st.write(f"Avg String Sim: {avg_scores['string']:.3f}")
            if 'semantic' in avg_scores:
                st.write(f"Avg Semantic Sim: {avg_scores['semantic']:.3f}")
            
            # LangChain metrics
            lc_results = [r for r in current_results if "langchain_evaluation" in r]
            if lc_results:
                st.write("**LangChain Metrics:**")
                lc_correctness_scores = []
                lc_relevance_scores = []
                lc_groundedness_scores = []
                lc_retrieval_scores = []
                
                for r in lc_results:
                    lc_eval = r["langchain_evaluation"]
                    if "correctness" in lc_eval and "score" in lc_eval["correctness"]:
                        lc_correctness_scores.append(lc_eval["correctness"]["score"])
                    if "relevance" in lc_eval and "score" in lc_eval["relevance"]:
                        lc_relevance_scores.append(lc_eval["relevance"]["score"])
                    if "groundedness" in lc_eval and "score" in lc_eval["groundedness"]:
                        lc_groundedness_scores.append(lc_eval["groundedness"]["score"])
                    if "retrieval_relevance" in lc_eval and "score" in lc_eval["retrieval_relevance"]:
                        lc_retrieval_scores.append(lc_eval["retrieval_relevance"]["score"])
                
                if lc_correctness_scores:
                    st.write(f"Avg Correctness: {np.mean(lc_correctness_scores):.1f}/5")
                if lc_relevance_scores:
                    st.write(f"Avg Relevance: {np.mean(lc_relevance_scores):.1f}/5")
                if lc_groundedness_scores:
                    st.write(f"Avg Groundedness: {np.mean(lc_groundedness_scores):.1f}/5")
                if lc_retrieval_scores:
                    st.write(f"Avg Retrieval Relevance: {np.mean(lc_retrieval_scores):.1f}/5")
                
                st.caption(f"LangChain evaluated: {len(lc_results)}/{len(current_results)} results")
            else:
                st.caption("No LangChain evaluations yet")
            
            # RAGAS metrics
            ragas_results = [r for r in current_results if "ragas_evaluation" in r]
            if ragas_results:
                st.write("**RAGAS Metrics:**")
                ragas_scores = {
                    "context_recall": [],
                    "context_precision": [],
                    # "context_entity_recall": [],
                    "faithfulness": [],
                    "factual_correctness": [],
                    "answer_accuracy": []
                }
                
                for r in ragas_results:
                    ragas_eval = r["ragas_evaluation"]
                    for metric_name, score_list in ragas_scores.items():
                        if metric_name in ragas_eval and "score" in ragas_eval[metric_name]:
                            score = ragas_eval[metric_name]["score"]
                            if isinstance(score, (int, float)) and not isinstance(score, bool):
                                score_list.append(score)
                
                # Display averages for metrics with data (5 core metrics)
                metric_display_names = {
                    "context_recall": "Context Recall",
                    "context_precision": "Context Precision", 
                    # "context_entity_recall": "Context Entity Recall",
                    "faithfulness": "Faithfulness",
                    "factual_correctness": "Factual Correctness",
                    "answer_accuracy": "Answer Accuracy"
                }
                
                for metric_name, scores in ragas_scores.items():
                    if scores:
                        avg_score = np.mean(scores)
                        display_name = metric_display_names[metric_name]
                        st.write(f"Avg {display_name}: {avg_score:.3f}")
                
                st.caption(f"RAGAS evaluated: {len(ragas_results)}/{len(current_results)} results")
            else:
                st.caption("No RAGAS evaluations yet")
        
        st.write(f"**Total Session:** {len(st.session_state.rag_evaluation_history)} evaluations")

# --- NEW: Show RAG Evaluation Results ---
def show_rag_evaluation_results(result):
    st.markdown("---")
    st.subheader("📊 RAG Q&A Evaluation Results")
    # Retrieved Context Section
    st.markdown("#### Retrieved Context")
    source_docs = result.get("source_docs", [])
    st.write(f"Retrieved {len(source_docs)} documents")
    if source_docs:
        meta_rows = []
        for i, meta in enumerate(source_docs):
            meta_rows.append({
                "Doc #": i + 1,
                "CELEX_ID": meta.get("celex", "N/A"),
                "Title": meta.get("case_title", "N/A"),
                "Date": meta.get("date", "N/A"),
                "Paragraph": meta.get("para_no", "N/A"),
                "Role": meta.get("role", "N/A")
            })
        st.table(meta_rows)
    context = result.get("retrieved_context", "")
    if context:
        st.text_area("Context from retrieved documents:", context, height=200)
    else:
        st.warning("No context was retrieved.")
    # Show answer
    st.markdown("#### Retrieved Answer (RAG)")
    st.info(result["generated_answer"])
    # Reference answer
    st.markdown("#### Reference Answer")
    st.info(result["reference_answer"])
    # Evaluation metrics
    show_evaluation_results(result)
    
    # Show LangChain evaluation if available
    if "langchain_evaluation" in result:
        show_langchain_evaluation_results(result["langchain_evaluation"])
    
    # Show RAGAS evaluation if available
    if "ragas_evaluation" in result:
        show_ragas_evaluation_results(result["ragas_evaluation"])
        
        # RAGAS evaluation model info
        with st.expander("🔧 RAGAS Configuration", expanded=False):
            ragas_provider = st.session_state.get("ragas_llm_provider")
            ragas_model = st.session_state.get("ragas_llm_model")
            
            if ragas_provider and ragas_model:
                st.write(f"**Evaluation LLM:** {ragas_provider}/{ragas_model}")
            
            # Show basic success/failure summary
            ragas_eval = result["ragas_evaluation"]
            successful_count = sum(1 for metric in ["context_recall", "context_precision", # "context_entity_recall", 
                                                  "faithfulness", "factual_correctness", "answer_accuracy"] 
                                 if metric in ragas_eval and ragas_eval[metric].get("score", 0) > 0 and "error" not in ragas_eval[metric])
            
            st.write(f"**Successful Metrics:** {successful_count}/5")

def _extract_numeric_score(score_value, default=0):
    """
    Safely extract numeric score from various formats.
    Handles cases where score might be a dict, string, or already numeric.
    """
    if isinstance(score_value, (int, float)):
        return score_value
    elif isinstance(score_value, dict):
        # If it's a dict, try to extract 'score' or 'value' field
        return score_value.get("score", score_value.get("value", default))
    elif isinstance(score_value, str):
        try:
            return float(score_value)
        except ValueError:
            return default
    else:
        return default

def show_langchain_evaluation_results(lc_eval: Dict[str, Any]):
    """Display LangChain RAG evaluation results"""
    st.markdown("---")
    st.subheader("🔬 LangChain RAG Evaluation Results")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        correctness = lc_eval.get("correctness", {})
        raw_score = correctness.get("score", 0)
        score = _extract_numeric_score(raw_score, 0)
        color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
        st.metric("Correctness", f"{score}/5")
        st.caption(f"{color} {_get_score_description(score, 'correctness')}")
    
    with col2:
        relevance = lc_eval.get("relevance", {})
        raw_score = relevance.get("score", 0)
        score = _extract_numeric_score(raw_score, 0)
        color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
        st.metric("Relevance", f"{score}/5")
        st.caption(f"{color} {_get_score_description(score, 'relevance')}")
    
    with col3:
        groundedness = lc_eval.get("groundedness", {})
        raw_score = groundedness.get("score", 0)
        score = _extract_numeric_score(raw_score, 0)
        color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
        st.metric("Groundedness", f"{score}/5")
        st.caption(f"{color} {_get_score_description(score, 'groundedness')}")
    
    with col4:
        retrieval_rel = lc_eval.get("retrieval_relevance", {})
        raw_score = retrieval_rel.get("score", 0)
        score = _extract_numeric_score(raw_score, 0)
        color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
        st.metric("Retrieval Relevance", f"{score}/5")
        st.caption(f"{color} {_get_score_description(score, 'retrieval_relevance')}")
    
    # Show explanations for debugging if needed
    with st.expander("📝 Evaluation Details", expanded=False):
        if correctness.get("explanation"):
            st.markdown("**Correctness Explanation:**")
            st.write(correctness.get("explanation", "No explanation available"))
        
        if relevance.get("explanation"):
            st.markdown("**Relevance Explanation:**")
            st.write(relevance.get("explanation", "No explanation available"))
        
        if groundedness.get("explanation"):
            st.markdown("**Groundedness Explanation:**")
            st.write(groundedness.get("explanation", "No explanation available"))
        
        if retrieval_rel.get("explanation"):
            st.markdown("**Retrieval Relevance Explanation:**")
            st.write(retrieval_rel.get("explanation", "No explanation available"))

def show_ragas_evaluation_results(ragas_eval: Dict[str, Any]):
    """Display RAGAS RAG evaluation results with 4 core metrics"""
    st.markdown("---")
    st.subheader("📊 RAGAS RAG Evaluation Results")
    
    # Define the 5 core metrics with their display information
    metrics_info = [
        ("context_recall", "Context Recall", "🔍"),
        ("context_precision", "Context Precision", "🎯"),
        # ("context_entity_recall", "Entity Recall", "🏷️"),
        ("faithfulness", "Faithfulness", "✅"),
        ("factual_correctness", "Factual Correctness", "📋"),
        ("answer_accuracy", "Answer Accuracy", "🎯")
    ]
    
    # Display 5 metrics across 2 rows: 3 metrics in first row, 2 in second row
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)
    cols = [col1, col2, col3, col4, col5]
    
    # Display all 5 metrics
    for i, (metric_key, metric_name, emoji) in enumerate(metrics_info):
        with cols[i]:
            metric_data = ragas_eval.get(metric_key, {})
            score = metric_data.get("score", 0.0)
            
            if "error" in metric_data:
                st.metric(f"{emoji} {metric_name}", "Error")
                st.caption("🔴 Evaluation failed")
            else:
                color = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
                st.metric(f"{emoji} {metric_name}", f"{score:.3f}")
                st.caption(f"{color} {_get_ragas_score_description(score, metric_key)}")
    
    # Metric descriptions in expander
    with st.expander("📚 RAGAS Metric Descriptions", expanded=False):
        st.markdown("**Context Recall:** Measures how much of the relevant information from the reference is captured in the retrieved context. Higher values indicate better retrieval completeness.")
        st.markdown("**Context Precision:** Measures the precision of retrieved context by evaluating what proportion of the retrieved context is relevant to answering the question.")
        # st.markdown("**Context Entity Recall:** Measures recall based on entities present in ground truth and context. Useful for entity-focused legal applications.")
        st.markdown("**Faithfulness:** Measures the factual consistency of the generated answer against the retrieved context. Higher values indicate less hallucination.")
        st.markdown("**Factual Correctness:** Measures the factual accuracy of the generated answer compared to the reference answer using high atomicity and coverage for detailed legal analysis.")
        st.markdown("**Answer Accuracy:** Measures the accuracy of the generated answer compared to the reference answer. Higher values indicate more accurate answers.")
        
        st.markdown("---")
        st.markdown("**📝 Note:** Noise Sensitivity and Response Relevancy metrics have been disabled to reduce LLM call overhead and prevent timeouts.")
        
        # Show FactualCorrectness configuration
        st.markdown("---")
        st.markdown("**FactualCorrectness Configuration:**")
        st.markdown("• Mode: F1 (balanced precision and recall)")
        st.markdown("• Atomicity: Low (Less detailed claim decomposition)")
        st.markdown("• Coverage: High (comprehensive evaluation)")
        st.markdown("• Optimized for legal Q&A systems")
        
        # Show any errors for the 4 core metrics
        for metric_key, metric_name, _ in metrics_info:
            if metric_key in ragas_eval and "error" in ragas_eval[metric_key]:
                st.error(f"**{metric_name} Error:** {ragas_eval[metric_key]['error']}")
    
    # Simplified debug information for troubleshooting
    with st.expander("🔧 RAGAS Debug Information", expanded=False):
        # Error summary
        metrics_with_errors = []
        zero_scores = []
        
        for metric_key, metric_name, emoji in metrics_info:
            if metric_key in ragas_eval:
                metric_data = ragas_eval[metric_key]
                score = metric_data.get("score", 0.0)
                
                if "error" in metric_data:
                    metrics_with_errors.append(f"{emoji} {metric_name}")
                elif score == 0.0:
                    zero_scores.append(f"{emoji} {metric_name}")
        
        # Simple error summary
        col_err1, col_err2 = st.columns(2)
        with col_err1:
            st.metric("Metrics with Errors", len(metrics_with_errors))
        with col_err2:
            st.metric("Zero Scores", len(zero_scores))
        
        # Show specific errors for failed metrics
        if metrics_with_errors:
            st.markdown("**❌ Failed Metrics:**")
            for metric_key, metric_name, emoji in metrics_info:
                if metric_key in ragas_eval and "error" in ragas_eval[metric_key]:
                    st.error(f"**{emoji} {metric_name}:** {ragas_eval[metric_key]['error']}")
        
        # Brief explanation for zero scores
        if zero_scores:
            st.markdown("**ℹ️ Zero Scores:**")
            st.info("Zero scores are common in legal Q&A systems where retrieved context may not contain all reference answer information. Focus on Faithfulness and Factual Correctness as the most reliable indicators.")

def _get_ragas_score_description(score: float, metric_type: str) -> str:
    """Get description for RAGAS score (0.0-1.0 scale) for the 5 core metrics"""
    if score >= 0.8:
        descriptions = {
            "context_recall": "Excellent Retrieval",
            "context_precision": "Highly Precise",
            # "context_entity_recall": "Excellent Entity Coverage",
            "faithfulness": "Highly Faithful",
            "factual_correctness": "Highly Accurate",
            "answer_accuracy": "Highly Accurate"
        }
        return descriptions.get(metric_type, "Excellent")
    elif score >= 0.6:
        descriptions = {
            "context_recall": "Good Retrieval",
            "context_precision": "Precise",
            # "context_entity_recall": "Good Entity Coverage",
            "faithfulness": "Faithful",
            "factual_correctness": "Accurate",
            "answer_accuracy": "Accurate"
        }
        return descriptions.get(metric_type, "Good")
    elif score >= 0.4:
        descriptions = {
            "context_recall": "Partial Retrieval",
            "context_precision": "Somewhat Precise",
            # "context_entity_recall": "Partial Entity Coverage",
            "faithfulness": "Somewhat Faithful",
            "factual_correctness": "Partially Accurate",
            "answer_accuracy": "Partially Accurate"
        }
        return descriptions.get(metric_type, "Fair")
    else:
        descriptions = {
            "context_recall": "Poor Retrieval",
            "context_precision": "Imprecise",
            # "context_entity_recall": "Poor Entity Coverage",
            "faithfulness": "Unfaithful",
            "factual_correctness": "Inaccurate",
            "answer_accuracy": "Inaccurate"
        }
        return descriptions.get(metric_type, "Poor")

def _get_score_description(score, metric_type: str) -> str:
    """Get description for score (handles both int and float)"""
    # Convert to numeric if needed
    if not isinstance(score, (int, float)):
        score = 0
    
    if score >= 4:
        if metric_type == "correctness":
            return "Excellent"
        elif metric_type == "relevance":
            return "Highly Relevant"
        elif metric_type == "groundedness":
            return "Well Grounded"
        elif metric_type == "retrieval_relevance":
            return "Highly Relevant"
        else:
            return "Excellent"
    elif score >= 3:
        if metric_type == "correctness":
            return "Good"
        elif metric_type == "relevance":
            return "Relevant"
        elif metric_type == "groundedness":
            return "Grounded"
        elif metric_type == "retrieval_relevance":
            return "Relevant"
        else:
            return "Good"
    elif score >= 2:
        if metric_type == "correctness":
            return "Fair"
        elif metric_type == "relevance":
            return "Somewhat Relevant"
        elif metric_type == "groundedness":
            return "Partially Grounded"
        elif metric_type == "retrieval_relevance":
            return "Somewhat Relevant"
        else:
            return "Fair"
    else:
        if metric_type == "correctness":
            return "Poor"
        elif metric_type == "relevance":
            return "Not Relevant"
        elif metric_type == "groundedness":
            return "Hallucinated"
        elif metric_type == "retrieval_relevance":
            return "Not Relevant"
        else:
            return "Poor"





def extract_citations(text):
    """
    Extracts all (CELEX_ID:PARA_NO) citations from the text.
    Returns a set of tuples: {(celex, para), ...}
    """
    pattern = r"\((\d{6,}CJ\d{4,}):(\d+)\)"
    return set(re.findall(pattern, text))

def context_citations(context):
    """
    Extracts all (CELEX_ID:PARA_NO) pairs from the context string.
    Returns a set of tuples: {(celex, para), ...}
    """
    return extract_citations(context)

def validate_citations(answer, context):
    """
    Checks if all citations in the answer are present in the context.
    Returns a list of invalid citations (those not found in context).
    """
    answer_cites = extract_citations(answer)
    context_cites = context_citations(context)
    invalid = answer_cites - context_cites
    return list(invalid)

def deduplicate_retrieved_docs(source_docs, target_count=None):
    """
    Remove duplicate paragraphs from retrieved documents.
    
    Only removes documents that have the EXACT same CELEX ID AND paragraph number.
    Different paragraphs from the same case are kept.
    
    For documents with same (CELEX, paragraph), keep only one instance,
    preferring 'cited' role over 'citing' role as cited paragraphs are 
    typically more authoritative/referenced.
    
    Args:
        source_docs: List of retrieved Document objects
        target_count: Optional target number of unique documents to return
        
    Returns:
        List of deduplicated Document objects
    """
    seen = {}  # (celex, para) -> Document
    duplicates_found = []  # Track what was deduplicated for debugging
    
    for doc in source_docs:
        celex = doc.metadata.get("celex", "")
        para = doc.metadata.get("para_no", "")
        role = doc.metadata.get("role", "")
        title = doc.metadata.get("case_title", "")
        
        key = (celex, para)  # Key is CELEX + paragraph number
        
        if key not in seen:
            # First time seeing this exact paragraph
            seen[key] = doc
        else:
            # We've seen this EXACT paragraph before (same CELEX + same paragraph)
            existing_role = seen[key].metadata.get("role", "")
            existing_title = seen[key].metadata.get("case_title", "")
            
            # Record this duplicate for debugging
            duplicates_found.append({
                "celex": celex,
                "paragraph": para,
                "title": title[:50] + "..." if len(title) > 50 else title,
                "existing_role": existing_role,
                "new_role": role,
                "action": "keep_cited" if role == "cited" and existing_role == "citing" else "keep_existing"
            })
            
            # Prefer 'cited' over 'citing' (cited paragraphs are more authoritative)
            if role == "cited" and existing_role == "citing":
                seen[key] = doc
            # If both are same role or existing is already 'cited', keep existing
    
    deduplicated = list(seen.values())
    
    # If target_count specified, return only that many documents
    if target_count and len(deduplicated) > target_count:
        deduplicated = deduplicated[:target_count]
    
    if len(deduplicated) < len(source_docs):
        removed_count = len(source_docs) - len(deduplicated)
        st.info(f"🔄 Removed {removed_count} duplicate paragraphs from retrieved context")
    
    return deduplicated

def smart_rag_retrieval(retriever, query: str, target_k: int = 10, expansion_factor: float = 1.8) -> List[Document]:
    """
    Smart RAG retrieval that handles deduplication properly.
    
    Retrieves more documents than needed, deduplicates, then returns target_k unique documents.
    This ensures you always get the requested number of unique documents.
    
    Args:
        retriever: The LangChain retriever object (any type)
        query: Query string
        target_k: Target number of unique documents to return
        expansion_factor: How many extra documents to retrieve initially (1.8 = 80% more)
        
    Returns:
        List of unique Document objects (exactly target_k documents if available)
    """
    # Safety check: ensure retriever is not None
    if retriever is None:
        st.error("❌ RAG retriever is not initialized. Please build the RAG system first.")
        return []
    
    # Calculate how many documents to retrieve initially
    initial_k = max(target_k, int(target_k * expansion_factor))
    
    # Handle different retriever types
    retriever_type = type(retriever).__name__
    
    try:
        if hasattr(retriever, 'search_kwargs'):
            # Standard vector retrievers (Chroma, etc.)
            original_k = retriever.search_kwargs.get("k", 10)
            retriever.search_kwargs["k"] = initial_k
            
            try:
                retrieved_docs = retriever.invoke(query)
            finally:
                # Restore original k value
                retriever.search_kwargs["k"] = original_k
                
        elif retriever_type == "ContextualCompressionRetriever":
            # Hybrid with reranking - handle the base retriever
            base_retriever = retriever.base_retriever
            
            if hasattr(base_retriever, 'search_kwargs'):
                # If base retriever has search_kwargs (shouldn't for ensemble, but just in case)
                original_k = base_retriever.search_kwargs.get("k", 10)
                base_retriever.search_kwargs["k"] = initial_k
                
                try:
                    retrieved_docs = retriever.invoke(query)
                finally:
                    base_retriever.search_kwargs["k"] = original_k
            else:
                # For ensemble retrievers, we can't easily modify k, so we'll retrieve with current settings
                # and rely on deduplication to handle the results
                st.info(f"🔄 Using hybrid retriever - retrieving with current settings and deduplicating")
                retrieved_docs = retriever.invoke(query)
                
        elif retriever_type == "EnsembleRetriever":
            # Hybrid without reranking - ensemble retrievers don't have simple k modification
            st.info(f"🔄 Using ensemble retriever - retrieving with current settings and deduplicating")
            retrieved_docs = retriever.invoke(query)
            
        else:
            # Unknown retriever type - try direct retrieval
            st.warning(f"⚠️ Unknown retriever type: {retriever_type}. Attempting direct retrieval.")
            retrieved_docs = retriever.invoke(query)
        
        # Deduplicate and return target number
        unique_docs = deduplicate_retrieved_docs(retrieved_docs, target_count=target_k)
        
        return unique_docs
        
    except Exception as retrieval_error:
        st.error(f"❌ Error during retrieval: {retrieval_error}")
        st.error(f"Retriever type: {retriever_type}")
        return []

# Export generic functions for reuse in other UI modules
__all__ = [
    'show_evaluation_results',
    'show_rag_evaluation_results', 
    'show_langchain_evaluation_results',
    'show_ragas_evaluation_results',
    'export_results',
    'smart_rag_retrieval',
    'run_langchain_rag_evaluation',
    'run_ragas_rag_evaluation',
    'display_results_preview',
    'display_detailed_analysis',
    'initialize_session_state',
    'deduplicate_retrieved_docs',
    'extract_citations',
    'context_citations',
    'validate_citations',
    '_extract_numeric_score',
    '_get_score_description',
    '_get_ragas_score_description'
]

if __name__ == "__main__":
    main()