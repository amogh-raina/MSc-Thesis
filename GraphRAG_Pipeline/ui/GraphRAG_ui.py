import sys
from pathlib import Path
import tempfile
import shutil
import os

# Add the project root and required paths to Python path
project_root = Path(__file__).resolve().parent.parent.parent
main_folder = project_root / "Main"
rag_ui_path = project_root / "RAG_Pipeline" / "ui"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(main_folder) not in sys.path:
    sys.path.insert(0, str(main_folder))
if str(rag_ui_path) not in sys.path:
    sys.path.insert(0, str(rag_ui_path))

import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="GraphRAG Legal Knowledge Evaluator",
    page_icon="📊",
    layout="wide"
)

import asyncio
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up tracing
os.environ["LANGCHAIN_TRACING_V2"] = "False"
os.environ["LANGCHAIN_PROJECT"] = "MSc_Thesis_GraphRAG"

# Import shared components from Main folder
from Main.core.model_manager import ModelManager, EmbeddingManager
from Main.core.question_bank import QuestionBank
from Main.core.evaluator import LLMEvaluator
from Main.config.settings import *

# Import generic functions from RAG_ui.py
from RAG_Pipeline.ui.RAG_ui import (
    show_evaluation_results,
    show_rag_evaluation_results, 
    show_langchain_evaluation_results,
    show_ragas_evaluation_results,
    export_results,
    run_langchain_rag_evaluation,
    run_ragas_rag_evaluation,
    display_results_preview,
    display_detailed_analysis,
    initialize_session_state,
    deduplicate_retrieved_docs,
    extract_citations,
    context_citations,
    validate_citations,
    _extract_numeric_score,
    _get_score_description,
    _get_ragas_score_description
)

# GraphRAG Pipeline imports
from GraphRAG_Pipeline.graph_pipeline import GraphRAGPipeline
from GraphRAG_Pipeline.GraphRAG_Prompt import _reuse_rag_prompt

def main():
    """Main GraphRAG Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    st.title("📊 GraphRAG Legal Knowledge Evaluator")
    st.markdown("Evaluate GraphRAG system performance with Neo4j graph database + vector embeddings")
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

    # GraphRAG-specific sidebar configuration
    selected_llm_provider, selected_llm_model, response_type = sidebar_configuration_graphrag()
    
    # Main interface
    if not st.session_state.question_bank_ready:
        st.info("👈 Please load the question bank from the sidebar to begin evaluation")
    elif st.session_state.rag_system is None:
        st.info("👈 Please upload dataset and build GraphRAG system from the sidebar")
    else:
        if selected_llm_provider and selected_llm_model:
            tab1, tab2 = st.tabs([
                "🔍 Manual GraphRAG Evaluation",
                "📊 Batch GraphRAG Evaluation"
            ])
            
            with tab1:
                manual_graphrag_evaluation_interface(selected_llm_provider, selected_llm_model, response_type)
                
            with tab2:
                batch_graphrag_evaluation_interface(selected_llm_provider, selected_llm_model, response_type)
        else:
            st.warning("Please select LLM provider and model from sidebar")


def sidebar_configuration_graphrag():
    """GraphRAG-specific sidebar configuration"""
    
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

        # GraphRAG Configuration
        with st.sidebar.expander("📊 GraphRAG Configuration"):
            st.markdown("Upload the **Paragraph-to-Paragraph** dataset (CSV) "
                        "and configure Neo4j + GraphRAG-specific settings.")

            uploaded_db = st.file_uploader("Dataset file", type=["csv"])
            
            # Neo4j Configuration
            st.markdown("**Neo4j Configuration**")
            neo4j_uri = st.text_input("Neo4j URI", "bolt://localhost:7687")
            neo4j_user = st.text_input("Neo4j Username", "neo4j")
            neo4j_password = st.text_input("Neo4j Password", type="password")

            # GraphRAG-specific embedding configuration
            st.markdown("**GraphRAG Embedding Configuration**")
            use_graphrag_embeddings = st.checkbox(
                "Enable GraphRAG Embeddings", 
                value=True,
                help="Enable embeddings for GraphRAG document retrieval",
                key="enable_graphrag_embeddings"
            )
            
            graphrag_emb_provider = None
            graphrag_emb_model = None
            if use_graphrag_embeddings and embedding_providers:
                graphrag_emb_provider = st.selectbox(
                    "GraphRAG Embedding Provider",
                    options=list(embedding_providers.keys()),
                    key="graphrag_emb_provider",
                    help="Select embedding provider for GraphRAG retrieval"
                )
                if graphrag_emb_provider:
                    graphrag_emb_model = st.selectbox(
                        "GraphRAG Embedding Model",
                        options=embedding_providers[graphrag_emb_provider]["models"],
                        key="graphrag_emb_model",
                        help="Select embedding model for GraphRAG retrieval"
                    )
            
            # GraphRAG Variant Selection
            st.markdown("**GraphRAG Variant Selection**")
            graphrag_variant = st.selectbox(
                "GraphRAG Variant",
                options=["Baseline", "Authority-weighted", "LangChain Reranker"],
                index=0,
                help="Choose GraphRAG ranking strategy:\n"
                     "• Baseline: Simple concatenation of vector + graph results\n"
                     "• Authority-weighted: Legal authority scoring with citation analysis\n"
                     "• LangChain Reranker: Advanced reranking with BGE/FlashRank/Jina models",
                key="graphrag_variant"
            )
            
            # Variant-specific parameters
            authority_weight = 0.3
            reranker_type = "cross_encoder"
            reranker_config = {}
            rerank_top_n = 15  # Default value for all variants
            
            if graphrag_variant == "Authority-weighted":
                st.markdown("**Authority Scoring Parameters**")
                authority_weight = st.slider(
                    "Authority Weight", 
                    0.1, 0.9, 0.3,
                    help="Balance between relevance (low) and authority (high)"
                )
                
                cited_by_weight = st.slider("Cited By Weight", 0.5, 2.0, 1.0)
                cites_weight = st.slider("Cites Weight", 0.5, 2.0, 0.8)
                case_authority_factor = st.slider("Case Authority Factor", 0.05, 0.3, 0.1)
                use_logarithmic_scaling = st.checkbox("Use Logarithmic Scaling", True)
                
                reranker_config = {
                    "cited_by_weight": cited_by_weight,
                    "cites_weight": cites_weight,
                    "case_authority_factor": case_authority_factor,
                    "use_logarithmic_scaling": use_logarithmic_scaling
                }
                
            elif graphrag_variant == "LangChain Reranker":
                st.markdown("**Reranker Configuration**")
                reranker_type = st.selectbox(
                    "Reranker Type",
                    options=["cross_encoder", "flashrank", "jina"],
                    index=0,
                    help="Choose reranker model:\n"
                         "• cross_encoder: BGE models (local/free)\n"
                         "• flashrank: Fast lightweight reranker\n"
                         "• jina: API-based reranker (requires API key)",
                    key="reranker_type"
                )
                
                rerank_top_n = st.number_input("Rerank Top N", 5, 30, 15)
                
                if reranker_type == "cross_encoder":
                    bge_model = st.selectbox(
                        "BGE Model",
                        options=["BAAI/bge-reranker-base", "BAAI/bge-reranker-large", "BAAI/bge-reranker-v2-m3"],
                        index=0,
                        key="bge_model"
                    )
                    reranker_config = {"model_name": bge_model}
                else:
                    reranker_config = {}
            
            # Retrieval settings
            st.markdown("**Retrieval Settings**")
            k_value = st.number_input(
                "Number of initial vector results (k)",
                min_value=5,
                max_value=50,
                value=10,
                help="Number of initial vector similarity results before graph expansion",
                key="graphrag_k_value"
            )
            
            expansion_k = st.number_input(
                "Graph expansion results (expansion_k)",
                min_value=5,
                max_value=100,
                value=20,
                help="Maximum additional results from graph expansion",
                key="graphrag_expansion_k"
            )
            
            max_total_results = st.number_input(
                "Maximum total results",
                min_value=10,
                max_value=100,
                value=30,
                help="Maximum total results to return after ranking",
                key="graphrag_max_total"
            )

            col_build, col_reset = st.columns(2)
            if col_build.button("🔧 Build GraphRAG", use_container_width=True, key="graphrag_build_btn"):
                if not uploaded_db:
                    st.error("Please upload a CSV file first.")
                elif not neo4j_uri or not neo4j_user or not neo4j_password:
                    st.error("Please provide all Neo4j connection details.")
                elif not use_graphrag_embeddings:
                    st.error("Please enable GraphRAG embeddings.")
                elif not graphrag_emb_provider or not graphrag_emb_model:
                    st.error("Please select both GraphRAG embedding provider and model.")
                else:
                    _setup_graphrag(
                        uploaded_db, neo4j_uri, neo4j_user, neo4j_password,
                        graphrag_emb_provider, graphrag_emb_model,
                        graphrag_variant, k_value, expansion_k, max_total_results,
                        authority_weight, reranker_type, reranker_config, rerank_top_n
                    )

            # Reset options
            col_reset1, col_reset2 = st.columns(2)
            if col_reset1.button("🗑 Reset", use_container_width=True, key="graphrag_reset_btn", 
                                help="Clear session state and reset Neo4j"):
                _reset_graphrag()
            
            if col_reset2.button("🔄 Soft Reset", use_container_width=True, key="graphrag_soft_reset_btn",
                                help="Clear session state only (keep Neo4j data)"):
                _soft_reset_graphrag()

        # LangChain RAG Evaluation Configuration (reuse from RAG_ui)
        with st.sidebar.expander("🔬 LangChain RAG Evaluation"):
            st.markdown("Configure LLM-as-a-Judge evaluation for GraphRAG responses using LangChain framework")
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

        # RAGAS RAG Evaluation Configuration (reuse from RAG_ui)
        with st.sidebar.expander("📊 RAGAS RAG Evaluation"):
            st.markdown("Configure mathematically-grounded GraphRAG evaluation using RAGAS metrics")
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

        # Question bank loading (reuse from RAG_ui)
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
        
        # Export section (reuse from RAG_ui)
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


def _setup_graphrag(file_obj, neo4j_uri, neo4j_user, neo4j_password,
                   emb_provider, emb_model, variant, k_value, expansion_k, max_total_results,
                   authority_weight, reranker_type, reranker_config, rerank_top_n=15):
    """Setup GraphRAG pipeline with selected variant"""
    
    # Save uploaded file to temporary path
    tmp_path = Path(tempfile.gettempdir()) / f"tmp_{file_obj.name}"
    tmp_path.write_bytes(file_obj.getbuffer())

    # Create embedding model
    embed_obj = EmbeddingManager.create_embedding_model(emb_provider, emb_model)
    if embed_obj is None:
        st.error("Failed to create GraphRAG embedding model.")
        return

    try:
        with st.spinner("🏗️ Building GraphRAG system..."):
            # Create GraphRAG pipeline
            pipeline = GraphRAGPipeline(
                dataset_path=tmp_path,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                embedding=embed_obj,
                force_rebuild_graph=False,
                force_rebuild_vectors=False,
                vector_index_name="graphrag_vector_index"
            )
            
            # Initialize pipeline (builds graph + vectors)
            pipeline.initialize()
            
            # Create appropriate retriever variant
            if variant == "Baseline":
                retriever = pipeline.get_baseline_retriever(
                    k=k_value, 
                    expansion_k=expansion_k, 
                    max_total_results=max_total_results
                )
            elif variant == "Authority-weighted":
                retriever = pipeline.get_authority_retriever(
                    k=k_value, 
                    expansion_k=expansion_k, 
                    max_total_results=max_total_results,
                    authority_weight=authority_weight,
                    **reranker_config
                )
            elif variant == "LangChain Reranker":
                retriever = pipeline.get_langchain_reranker_retriever(
                    k=k_value, 
                    expansion_k=expansion_k, 
                    max_total_results=max_total_results,
                    reranker_type=reranker_type,
                    rerank_top_n=rerank_top_n,
                    reranker_config=reranker_config
                )
            
            # Store in session state
            st.session_state.rag_system = pipeline
            st.session_state.rag_retriever = retriever
            st.session_state.rag_title_idx = None  # GraphRAG doesn't use title index
            
            # Store GraphRAG configuration
            st.session_state.rag_config = {
                "embedding_provider": emb_provider,
                "embedding_model": emb_model,
                "variant": variant,
                "k_value": k_value,
                "expansion_k": expansion_k,
                "max_total_results": max_total_results,
                "authority_weight": authority_weight,
                "reranker_type": reranker_type,
                "reranker_config": reranker_config,
                "rerank_top_n": rerank_top_n,
                "neo4j_uri": neo4j_uri,
                "neo4j_user": neo4j_user,
                "pipeline_type": "GraphRAG"
            }
            
            # Display success message
            st.success(f"✅ GraphRAG ready! Variant: {variant}, K={k_value}, Expansion={expansion_k}")
            
            # Display pipeline stats
            try:
                stats = pipeline.get_pipeline_status()
                if stats.get("vectors_ready") and stats.get("graph_ready"):
                    st.info(f"📊 Graph + Vector store ready for {variant} retrieval")
                else:
                    st.warning("⚠️ Pipeline partially ready - check Neo4j connection")
            except Exception as stats_error:
                st.warning(f"Could not retrieve pipeline stats: {stats_error}")
    
    except Exception as e:
        st.error(f"❌ Error setting up GraphRAG: {str(e)}")
        st.error(f"🔍 Error details: {type(e).__name__}: {str(e)}")


def _reset_graphrag():
    """Reset GraphRAG system by clearing session state"""
    try:
        # Close GraphRAG pipeline connections
        if st.session_state.get("rag_system"):
            try:
                st.session_state.rag_system.close()
            except Exception as e:
                st.warning(f"Warning during GraphRAG cleanup: {e}")
        
        # Clear session state references
        st.session_state.rag_system = None
        st.session_state.rag_retriever = None
        st.session_state.rag_title_idx = None
        if "rag_config" in st.session_state:
            del st.session_state.rag_config
        
        st.success("✅ GraphRAG system reset!")
        st.info("💡 Note: Neo4j data remains. Use Neo4j Browser to clear database if needed.")
            
    except Exception as e:
        # Clear session state even if cleanup fails
        st.session_state.rag_system = None
        st.session_state.rag_retriever = None
        st.session_state.rag_title_idx = None
        if "rag_config" in st.session_state:
            del st.session_state.rag_config
        st.error(f"❌ Error during reset: {str(e)}")


def _soft_reset_graphrag():
    """Soft reset - clear session state only, keep Neo4j data"""
    try:
        # Clear session state references
        st.session_state.rag_system = None
        st.session_state.rag_retriever = None
        st.session_state.rag_title_idx = None
        if "rag_config" in st.session_state:
            del st.session_state.rag_config
        
        st.success("✅ GraphRAG session state cleared! Neo4j data preserved.")
        st.info("💡 Use 'Build GraphRAG' to reconnect to existing data or upload new dataset.")
        
    except Exception as e:
        st.error(f"❌ Error during soft reset: {str(e)}")


def manual_graphrag_evaluation_interface(selected_llm_provider, selected_llm_model, response_type):
    """Manual GraphRAG evaluation interface - reuses most logic from RAG_ui"""
    
    st.subheader("🔍 Manual GraphRAG Evaluation")
    
    question = st.text_area(
        "Enter your legal question for GraphRAG evaluation:",
        placeholder="e.g., What are the consequences of failing to implement EU directives?",
        height=100
    )
    
    # Reference answer handling (same as RAG_ui)
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
        
    if st.button("📊 Generate GraphRAG Answer & Evaluate", type="primary"):
        if not question.strip():
            st.error("❌ Please provide a question")
        elif reference_mode == "✏️ Provide manually" and not manual_reference_answer.strip():
            st.error("❌ Please provide a reference answer")
        else:
            if st.session_state.rag_retriever is None:
                st.error("❌ GraphRAG system is not initialized. Please build the GraphRAG system first.")
                return
            
            with st.spinner("📊 Generating GraphRAG answer and evaluating..."):
                # Generate GraphRAG answer using direct retriever call
                llm = ModelManager.create_llm(selected_llm_provider, selected_llm_model)
                
                # Use GraphRAG retriever directly (handles its own expansion and deduplication)
                source_docs = st.session_state.rag_retriever.invoke(question)
                
                # Create context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in source_docs])
                
                # Generate answer using GraphRAG prompt
                from langchain.prompts import PromptTemplate, ChatPromptTemplate
                prompt_template = _reuse_rag_prompt(response_type)
                
                # Handle both ChatPromptTemplate and PromptTemplate
                if isinstance(prompt_template, ChatPromptTemplate):
                    formatted_prompt = prompt_template.format(question=question, context=context)
                elif hasattr(prompt_template, 'template'):
                    template_str = prompt_template.template
                    formatted_prompt = template_str.format(question=question, context=context)
                else:
                    template_str = str(prompt_template)
                    formatted_prompt = template_str.format(question=question, context=context)
                
                answer = llm.invoke(formatted_prompt).content
                
                # Get reference answer (same logic as RAG_ui)
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
                    # Evaluate the GraphRAG answer (reuse from RAG_ui)
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
                        "graphrag": True,
                        "timestamp": datetime.now().isoformat(),
                        "retrieved_context": context,
                        "source_docs": [doc.metadata for doc in source_docs],
                        "rag_config": st.session_state.get("rag_config", {}),
                    }
                    
                    # Run additional evaluations if enabled (reuse from RAG_ui)
                    result = run_langchain_rag_evaluation([result])[0] if st.session_state.get("enable_rag_eval", False) else result
                    result = run_ragas_rag_evaluation([result])[0] if st.session_state.get("enable_ragas_eval", False) else result
                    
                    # Store in evaluation history
                    st.session_state.rag_evaluation_history.append(result)
                    st.session_state.evaluation_history.append(result)
                    
                    # Confirm storage
                    total_stored = len(st.session_state.evaluation_history)
                    st.success(f"✅ GraphRAG evaluation completed! ({total_stored} total results stored)")
                    
                    # Show results (reuse from RAG_ui)
                    show_rag_evaluation_results(result)
                    
                    # Add immediate export options
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


def batch_graphrag_evaluation_interface(selected_llm_provider, selected_llm_model, response_type):
    """Batch GraphRAG evaluation interface - reuses most logic from RAG_ui"""
    
    st.subheader("📊 Batch GraphRAG Evaluation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Evaluate multiple questions automatically using GraphRAG system with database questions and answers")
        
        max_questions = st.number_input(
            "Maximum questions to evaluate:",
            min_value=1,
            max_value=100,
            value=10,
            help="Limit the number of questions for testing"
        )
        
        if st.button("🚀 Start Batch GraphRAG Evaluation", type="primary"):
            with st.spinner("Running batch GraphRAG evaluation..."):
                # Get questions from database
                questions = st.session_state.evaluator.question_bank.get_all_questions()
                questions = questions[:max_questions]
                results = []
                
                # Prepare for batch GraphRAG evaluation
                llm = ModelManager.create_llm(selected_llm_provider, selected_llm_model)
                
                # Prepare prompt template
                from langchain.prompts import PromptTemplate, ChatPromptTemplate
                prompt_template = _reuse_rag_prompt(response_type)
                
                # Process each question
                progress_bar = st.progress(0)
                for idx, qdata in enumerate(questions):
                    question_text = qdata["question_text"]
                    reference_answer = qdata["answer_text"]
                    
                    try:
                        # Generate GraphRAG answer using direct retriever call
                        source_docs = st.session_state.rag_retriever.invoke(question_text)
                        
                        # Create context and generate answer
                        context = "\n\n".join([doc.page_content for doc in source_docs])
                        
                        # Format prompt based on template type
                        if isinstance(prompt_template, ChatPromptTemplate):
                            formatted_prompt = prompt_template.format(question=question_text, context=context)
                        elif hasattr(prompt_template, 'template'):
                            template_str = prompt_template.template
                            formatted_prompt = template_str.format(question=question_text, context=context)
                        else:
                            template_str = str(prompt_template)
                            formatted_prompt = template_str.format(question=question_text, context=context)
                        
                        answer = llm.invoke(formatted_prompt).content
                        
                        # Evaluate answer (reuse from RAG_ui)
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
                            "graphrag": True,
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
                
                # Run additional evaluations if enabled (reuse from RAG_ui)
                if st.session_state.get("enable_rag_eval", False) and results:
                    results = run_langchain_rag_evaluation(results)
                
                if st.session_state.get("enable_ragas_eval", False) and results:
                    results = run_ragas_rag_evaluation(results)
                
                if results:
                    # Store results in both histories
                    st.session_state.rag_evaluation_history.extend(results)
                    st.session_state.evaluation_history.extend(results)
                    
                    # Calculate aggregate scores (reuse from RAG_ui)
                    aggregate_scores = st.session_state.evaluator.calculate_aggregate_scores(results)
                    
                    st.success(f"✅ Batch GraphRAG evaluation completed! Processed {len(results)} questions")
                    
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
                    
                    # Results table preview (reuse from RAG_ui)
                    st.subheader("📋 Results Preview")
                    display_results_preview(results)
                    
                    # Export note
                    st.info("💡 **Export Results:** Export buttons are available below and in the sidebar.")
                    
                    # Detailed analysis in expander (reuse from RAG_ui)
                    with st.expander("📊 Detailed Analysis", expanded=False):
                        display_detailed_analysis(results, selected_llm_provider, selected_llm_model, response_type)
                    
                    # Add immediate export options
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
                    st.error("❌ Batch GraphRAG evaluation failed - no results generated")
    
    with col2:
        # Display GraphRAG batch sidebar info
        display_graphrag_batch_sidebar_info(selected_llm_provider, selected_llm_model)


def display_graphrag_batch_sidebar_info(selected_llm_provider, selected_llm_model):
    """Display sidebar info for batch GraphRAG evaluation"""
    st.subheader("⚙️ Batch Settings (GraphRAG)")
    
    # GraphRAG System Status
    if st.session_state.rag_system and st.session_state.get("rag_config"):
        rag_config = st.session_state.rag_config
        st.success("🟢 GraphRAG System Ready")
        st.write("**GraphRAG Configuration:**")
        
        # Variant info
        variant = rag_config.get('variant', 'Unknown')
        st.write(f"• Variant: {variant}")
        
        # Parameters
        k_value = rag_config.get('k_value', 'N/A')
        expansion_k = rag_config.get('expansion_k', 'N/A')
        max_total = rag_config.get('max_total_results', 'N/A')
        st.write(f"• K-values: {k_value} → {expansion_k} → {max_total}")
        
        # Embedding info
        emb_provider = rag_config.get('embedding_provider', 'N/A')
        emb_model = rag_config.get('embedding_model', 'N/A')
        st.write(f"• Embedding: {emb_provider}/{emb_model}")
        
        # Variant-specific info
        if variant == "Authority-weighted":
            auth_weight = rag_config.get('authority_weight', 'N/A')
            st.write(f"• Authority Weight: {auth_weight}")
        elif variant == "LangChain Reranker":
            rerank_type = rag_config.get('reranker_type', 'N/A')
            st.write(f"• Reranker: {rerank_type}")
        
        st.info("🔗 **Neo4j + Vector** - Graph relationships + semantic search")
            
    elif st.session_state.rag_system:
        st.info("🟡 GraphRAG System Loaded (Legacy Config)")
    else:
        st.warning("🔴 GraphRAG System Not Ready")
    
    # Statistics about available questions
    if st.session_state.question_bank_ready:
        total_questions = len(st.session_state.evaluator.question_bank.questions)
        st.metric("Available Questions", total_questions)
        st.info("🟢 Database Ready")
    
    # Current session results
    if st.session_state.rag_evaluation_history:
        st.markdown("---")
        st.subheader("📊 Session Results (GraphRAG)")
        
        # Filter current model results
        current_results = [r for r in st.session_state.rag_evaluation_history 
                          if r.get("llm_provider") == selected_llm_provider 
                          and r.get("llm_model") == selected_llm_model
                          and r.get("graphrag", False)]
        
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
                # Calculate averages for LangChain metrics
                lc_scores = {}
                for metric in ["correctness", "relevance", "groundedness", "retrieval_relevance"]:
                    scores = []
                    for r in lc_results:
                        if metric in r["langchain_evaluation"] and "score" in r["langchain_evaluation"][metric]:
                            score = r["langchain_evaluation"][metric]["score"]
                            if isinstance(score, (int, float)):
                                scores.append(score)
                    if scores:
                        lc_scores[metric] = np.mean(scores)
                
                if lc_scores:
                    for metric, avg_score in lc_scores.items():
                        st.write(f"Avg {metric.title()}: {avg_score:.1f}/5")
                
                st.caption(f"LangChain evaluated: {len(lc_results)}/{len(current_results)} results")
            
            # RAGAS metrics
            ragas_results = [r for r in current_results if "ragas_evaluation" in r]
            if ragas_results:
                st.write("**RAGAS Metrics:**")
                ragas_scores = {}
                for metric in ["context_recall", "context_precision", "faithfulness", "factual_correctness", "answer_accuracy"]:
                    scores = []
                    for r in ragas_results:
                        if metric in r["ragas_evaluation"] and "score" in r["ragas_evaluation"][metric]:
                            score = r["ragas_evaluation"][metric]["score"]
                            if isinstance(score, (int, float)) and not isinstance(score, bool):
                                scores.append(score)
                    if scores:
                        ragas_scores[metric] = np.mean(scores)
                
                if ragas_scores:
                    for metric, avg_score in ragas_scores.items():
                        display_name = metric.replace("_", " ").title()
                        st.write(f"Avg {display_name}: {avg_score:.3f}")
                
                st.caption(f"RAGAS evaluated: {len(ragas_results)}/{len(current_results)} results")
        
        st.write(f"**Total Session:** {len(st.session_state.rag_evaluation_history)} evaluations")


if __name__ == "__main__":
    main()