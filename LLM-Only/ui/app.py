import sys
from pathlib import Path

# --- IMPORTANT ---
# This adds the project root directory (Integration) and the workspace root to the Python path.
# It allows the app to find modules from `core`, `utils`, `RAG_Pipeline`, and `Agent`.
# Do not remove this block.
integration_root = Path(__file__).resolve().parent.parent
workspace_root = integration_root.parent
if str(integration_root) not in sys.path:
    sys.path.insert(0, str(integration_root))
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))
# --- END IMPORTANT ---

import streamlit as st

# --- Page Configuration ---
# This MUST be the first Streamlit command used in an app, and must only be
# set once. We put it here, right after the streamlit import.
st.set_page_config(
    page_title="LLM Legal Knowledge Evaluator",
    page_icon="⚖️",
    layout="wide"
)

import os
from pathlib import Path
import tempfile
from io import BytesIO
import shutil
import asyncio
import json
import pandas as pd
import numpy as np
import regex as re
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up tracing
os.environ["LANGCHAIN_TRACING_V2"] = "False"
os.environ["LANGCHAIN_PROJECT"] = "MSc_Thesis"
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Import your modular components
from Main.core.model_manager import ModelManager, EmbeddingManager
from Main.core.question_bank import QuestionBank
from Main.core.evaluator import LLMEvaluator
from Main.config.settings import *

# RAG Pipeline imports
from RAG_Pipeline.rag_pipeline import RAGPipeline
from RAG_Pipeline.title_index import TitleIndex
from RAG_Pipeline.Langchain_eval_framework import LangchainEvalPipeline
from RAG_Pipeline.RAGAS_eval_framework import RAGASEvalPipeline

# Agent imports
from Agent.core.agent import LegalQAAgent
from Agent.tools.web_search_tool import WebSearchTool
from Agent.config.agent_config import AgentConfig

# GraphRAG integration in app.py
def _setup_graphrag(file_obj, neo4j_credentials):
    # Get GraphRAG-specific embedding from UI selection
    provider = st.session_state.get("graphrag_emb_provider")
    modelname = st.session_state.get("graphrag_emb_model")
    
    if not provider or not modelname:
        st.error("Please select GraphRAG embedding provider and model first")
        return
    
    embed_obj = EmbeddingManager.create_embedding_model(provider, modelname)
    if embed_obj is None:
        st.error("Failed to create GraphRAG embedding model")
        return
    
    try:
        # Save uploaded file to temp path
        import tempfile
        from pathlib import Path
        
        tmp_path = Path(tempfile.gettempdir()) / f"graphrag_{file_obj.name}"
        tmp_path.write_bytes(file_obj.getbuffer())
        
        # Create GraphRAG pipeline using the GraphRAG-specific embedding model
        from GraphRAG_Pipeline.graph_pipeline import create_graphrag_pipeline
        
        with st.spinner("Building GraphRAG system..."):
            pipeline = create_graphrag_pipeline(
                dataset_path=tmp_path,
                embedding=embed_obj,  # GraphRAG-specific embedding model
                neo4j_uri=neo4j_credentials["uri"],
                neo4j_user=neo4j_credentials["user"], 
                neo4j_password=neo4j_credentials["password"],
                auto_initialize=True
            )
        
        st.session_state.graphrag_system = pipeline
        st.success(f"✅ GraphRAG ready with {provider}/{modelname} embeddings!")
        
        # Display build summary
        status = pipeline.get_pipeline_status()
        st.info(f"📊 Built {status.get('total_paragraphs', 0)} paragraph nodes with vector embeddings")
        
    except Exception as e:
        st.error(f"❌ GraphRAG setup failed: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

def _reset_graphrag():
    """Reset the GraphRAG system"""
    if "graphrag_system" in st.session_state:
        try:
            # Close any open connections
            if hasattr(st.session_state.graphrag_system, 'close'):
                st.session_state.graphrag_system.close()
        except Exception as e:
            st.warning(f"Warning during GraphRAG cleanup: {e}")
        
        # Clear from session state
        del st.session_state.graphrag_system
    
    st.info("GraphRAG system reset.")

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    st.title("⚖️ LLM Legal Knowledge Evaluator")
    st.markdown("Evaluate LLM performance on legal questions")
    st.markdown("---")
    
    # Initialize evaluator
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = LLMEvaluator()
        st.session_state.evaluation_history = []
        st.session_state.question_bank_ready = False
        st.session_state.embedding_enabled = False
    if "rag_evaluation_history" not in st.session_state:
        st.session_state.rag_evaluation_history = []

    # Sidebar configuration
    selected_llm_provider, selected_llm_model, response_type = sidebar_configuration()
    
    # Tabs: Evaluation (Manual+Batch), RAG Q/A (Manual+Batch), Agent Q/A
    tab_eval, tab_rag, tab_agent = st.tabs([
        "📝 Evaluation (LLM)",
        "🔍 RAG Q&A", 
        "🤖 Agent Q&A"
    ])

    with tab_eval:
        if not st.session_state.question_bank_ready:
            st.info("👆 Please load the question bank from the sidebar to begin evaluation")
        else:
            if selected_llm_provider and selected_llm_model:
                # Manual Evaluation
                manual_evaluation_interface(selected_llm_provider, selected_llm_model, response_type)
                st.markdown("---")
                # Batch Evaluation
                batch_evaluation_interface(selected_llm_provider, selected_llm_model, response_type)
            else:
                st.warning("Please select LLM provider and model from sidebar")

    with tab_rag:
        rag_evaluation_interface(selected_llm_provider, selected_llm_model, response_type)

    with tab_agent:
        agent_qa_interface(selected_llm_provider, selected_llm_model, response_type)


def initialize_session_state():
    """Initialize session state variables similar to rag_app.py"""
    # RAG system state
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
        st.session_state.rag_retriever = None
        st.session_state.rag_title_idx = None

    # Agent system state
    if "agent_system" not in st.session_state:
        st.session_state.agent_system = None
    if "agent_ready" not in st.session_state:
        st.session_state.agent_ready = False
    if "agent_qa_history" not in st.session_state:
        st.session_state.agent_qa_history = []

    # Evaluation state
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = LLMEvaluator()
        st.session_state.evaluation_history = []
        st.session_state.question_bank_ready = False
        st.session_state.embedding_enabled = False
    if "rag_evaluation_history" not in st.session_state:
        st.session_state.rag_evaluation_history = []


def sidebar_configuration():
    """Sidebar configuration"""
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data directory input
        st.subheader("📄 Question Database")
        data_dir = st.text_input(
            "Data Directory Path", 
            value="./JSON Trial 1",
            help="Path to directory containing BEUL_EXAM_*.json files"
        )
        
        # Model Selection - MOVE THIS UP BEFORE AGENT CONFIGURATION
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
        
        # Embedding Configuration (only show if needed)
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
            "Enable embeddings for semantic-similarity & RAG",
            help="Toggle to compute the semantic similarity metric",
            key="enable_emb_sem"
        )

        # — Manual matching model selector —
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

        # — Semantic similarity model selector —
        emb_provider_sem = None
        emb_model_name_sem = None
        if use_embeddings_sem and embedding_providers:
            emb_provider_sem = st.selectbox(
                "Embedding Provider (semantic & RAG)",
                options=list(embedding_providers.keys()),
                key="emb_provider_sem"
            )
            if emb_provider_sem:
                emb_model_name_sem = st.selectbox(
                    "Embedding Model (semantic & RAG)",
                    options=embedding_providers[emb_provider_sem]["models"],
                    key="emb_model_name_sem"
                )

        with st.sidebar.expander("📚 RAG Configuration"):
            st.markdown("Upload the **Paragraph-to-Paragraph** dataset (CSV or JSON) "
                        "and pick the *same* embedding model you enabled above.")

            # data
            uploaded_db = st.file_uploader("Dataset file", type=["csv", "json", "jsonl"])

            # persist dir
            chroma_dir  = st.text_input("Chroma directory", "./chroma_db")

            # build / rebuild buttons
            col_build, col_reset = st.columns(2)
            if col_build.button("🔧 Build / Load", use_container_width=True, key="rag_build_btn"):
                if not uploaded_db:
                    st.error("Please upload a CSV/JSON first.")
                elif not st.session_state.get("enable_emb_sem", False):
                    st.error("Please enable a **Semantic & RAG** embedding in the Embedding Configuration above.")
                else:
                    _setup_rag(uploaded_db, chroma_dir)

            if col_reset.button("🗑 Reset", use_container_width=True, key="rag_reset_btn"):
                _reset_rag(chroma_dir)

        # Agent Configuration - NOW AFTER MODEL SELECTION
        with st.sidebar.expander("🤖 Agent Configuration"):
            st.markdown("Configure the multi-stage agentic RAG system")
            
            # Web search toggle
            enable_agent_web_search = st.checkbox(
                "Enable Web Search (Tavily)", 
                key="agent_web_search",
                help="Enable Tavily web search for Stage 3 retrieval"
            )
            
            # Agent thresholds
            st.markdown("**Retrieval Thresholds:**")
            vector_threshold = st.slider("Vector Store Threshold", 0.0, 1.0, 0.6, 0.1, key="vector_thresh")
            dataset_threshold = st.slider("Dataset Threshold", 0.0, 1.0, 0.7, 0.1, key="dataset_thresh")
            
            # Build agent button
            col_agent_build, col_agent_reset = st.columns(2)
            
            if col_agent_build.button("🔧 Build Agent", use_container_width=True, key="agent_build_btn"):
                if not st.session_state.get("rag_system"):
                    st.error("Please build RAG system first (above)")
                elif not st.session_state.get("enable_emb_sem", False):
                    st.error("Please enable semantic embeddings first")
                elif not selected_llm_provider or not selected_llm_model:
                    st.error("Please select an LLM provider and model first")
                else:
                    _setup_agent(enable_agent_web_search, vector_threshold, dataset_threshold, 
                               selected_llm_provider, selected_llm_model)
            
            if col_agent_reset.button("🗑 Reset Agent", use_container_width=True, key="agent_reset_btn"):
                _reset_agent()

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
        
        # Fixed Export section
        st.subheader("📊 Export Results")
        
        if st.session_state.evaluation_history:
            st.metric("Evaluations Completed", len(st.session_state.evaluation_history))
            
            export_format = st.selectbox("Export Format", ["excel", "json"])
            
            if st.button("📥 Export Results"):
                export_results(st.session_state.evaluation_history, export_format)
        else:
            st.text("No evaluations to export yet")
        
        return selected_llm_provider, selected_llm_model, response_type
        


def sidebar_configuration():
    """Sidebar configuration - main interface for settings"""
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data directory input
        st.subheader("📄 Question Database")
        data_dir = st.text_input(
            "Data Directory Path", 
            value="./JSON Trial 1",
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
            "Enable embeddings for semantic-similarity & RAG",
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
                "Embedding Provider (semantic & RAG)",
                options=list(embedding_providers.keys()),
                key="emb_provider_sem"
            )
            if emb_provider_sem:
                emb_model_name_sem = st.selectbox(
                    "Embedding Model (semantic & RAG)",
                    options=embedding_providers[emb_provider_sem]["models"],
                    key="emb_model_name_sem"
                )

        # RAG Configuration
        with st.sidebar.expander("📚 RAG Configuration"):
            st.markdown("Upload the **Paragraph-to-Paragraph** dataset (CSV or JSON) "
                        "and pick the *same* embedding model you enabled above.")

            uploaded_db = st.file_uploader("Dataset file", type=["csv", "json", "jsonl"])
            chroma_dir  = st.text_input("Chroma directory", "./chroma_db")

            col_build, col_reset = st.columns(2)
            if col_build.button("🔧 Build / Load", use_container_width=True, key="rag_build_btn"):
                if not uploaded_db:
                    st.error("Please upload a CSV/JSON first.")
                elif not st.session_state.get("enable_emb_sem", False):
                    st.error("Please enable a **Semantic & RAG** embedding in the Embedding Configuration above.")
                else:
                    _setup_rag(uploaded_db, chroma_dir)

            if col_reset.button("🗑 Reset", use_container_width=True, key="rag_reset_btn"):
                _reset_rag(chroma_dir)

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
                help="Enable 4-metric evaluation: context recall, faithfulness, factual correctness, response relevancy"
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
                
                # Embeddings for response relevancy (optional)
                ragas_use_embeddings = st.checkbox(
                    "Enable Response Relevancy (requires embeddings)",
                    key="ragas_use_embeddings",
                    help="Enable response relevancy metric - requires embeddings model"
                )
                
                ragas_emb_provider = None
                ragas_emb_model = None
                if ragas_use_embeddings and embedding_providers:
                    ragas_emb_provider = st.selectbox(
                        "RAGAS Embedding Provider",
                        options=list(embedding_providers.keys()),
                        key="ragas_emb_provider",
                        help="Select embedding provider for response relevancy"
                    )
                    if ragas_emb_provider:
                        ragas_emb_model = st.selectbox(
                            "RAGAS Embedding Model",
                            options=embedding_providers[ragas_emb_provider]["models"],
                            key="ragas_emb_model",
                            help="Select embedding model for response relevancy"
                        )
                
                # Display status
                if ragas_provider and ragas_model:
                    st.success(f"✅ RAGAS LLM: {ragas_provider}/{ragas_model}")
                    if ragas_use_embeddings and ragas_emb_provider and ragas_emb_model:
                        st.success(f"✅ RAGAS Embeddings: {ragas_emb_provider}/{ragas_emb_model}")
                    elif ragas_use_embeddings:
                        st.warning("Please select embedding provider and model for response relevancy")
                else:
                    st.warning("Please select both provider and model for RAGAS evaluation")
            else:
                # Clear RAGAS evaluation settings when disabled
                for key in ["ragas_llm_provider", "ragas_llm_model", "ragas_emb_provider", "ragas_emb_model"]:
                    if key in st.session_state:
                        del st.session_state[key]

        # GraphRAG Configuration
        with st.sidebar.expander("🕸️ GraphRAG Configuration"):
            st.markdown("Build a **Graph-RAG** system using Neo4j and vector embeddings")
            st.caption("📊 Creates citation graphs with semantic search capabilities")
            
            # Neo4j credentials (using environment variables as defaults)
            st.markdown("**Neo4j Connection:**")
            neo4j_uri = st.text_input("Neo4j URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"), key="graphrag_neo4j_uri")
            neo4j_user = st.text_input("Neo4j Username", value=os.getenv("NEO4J_USERNAME", "neo4j"), key="graphrag_neo4j_user")
            neo4j_password = st.text_input("Neo4j Password", value=os.getenv("NEO4J_PASSWORD", ""), type="password", key="graphrag_neo4j_password")
            
            # GraphRAG-specific embedding model selection
            st.markdown("**GraphRAG Embedding Model:**")
            graphrag_use_embeddings = st.checkbox(
                "Enable GraphRAG embeddings",
                key="enable_graphrag_embeddings",
                help="Select embedding model specifically for GraphRAG vector indexing"
            )
            
            graphrag_emb_provider = None
            graphrag_emb_model = None
            if graphrag_use_embeddings and embedding_providers:
                graphrag_emb_provider = st.selectbox(
                    "GraphRAG Embedding Provider",
                    options=list(embedding_providers.keys()),
                    key="graphrag_emb_provider",
                    help="Select embedding provider for GraphRAG"
                )
                if graphrag_emb_provider:
                    graphrag_emb_model = st.selectbox(
                        "GraphRAG Embedding Model", 
                        options=embedding_providers[graphrag_emb_provider]["models"],
                        key="graphrag_emb_model",
                        help="Select embedding model for GraphRAG vector indexing"
                    )
            
            # Dataset upload for GraphRAG
            st.markdown("**GraphRAG Dataset:**")
            uploaded_graphrag_dataset = st.file_uploader(
                "Legal citation dataset", 
                type=["csv", "json", "jsonl"],
                key="graphrag_dataset_upload",
                help="Upload paragraph-to-paragraph citation dataset for graph construction"
            )
            
            # Build/Reset buttons
            col_graphrag_build, col_graphrag_reset = st.columns(2)
            
            if col_graphrag_build.button("🔧 Build GraphRAG", use_container_width=True, key="graphrag_build_btn"):
                if not uploaded_graphrag_dataset:
                    st.error("Please upload a GraphRAG dataset first.")
                elif not graphrag_use_embeddings:
                    st.error("Please enable GraphRAG embeddings first.")
                elif not graphrag_emb_provider or not graphrag_emb_model:
                    st.error("Please select GraphRAG embedding provider and model.")
                elif not neo4j_uri or not neo4j_user or not neo4j_password:
                    st.error("Please provide Neo4j connection details.")
                else:
                    _setup_graphrag(uploaded_graphrag_dataset, {
                        "uri": neo4j_uri,
                        "user": neo4j_user,
                        "password": neo4j_password
                    })
            
            if col_graphrag_reset.button("🗑 Reset GraphRAG", use_container_width=True, key="graphrag_reset_btn"):
                _reset_graphrag()
            
            # Display GraphRAG status
            if st.session_state.get("graphrag_system"):
                st.success("✅ GraphRAG System Ready")
                graphrag_status = st.session_state.graphrag_system.get_pipeline_status()
                st.caption(f"Graph: {graphrag_status.get('graph_ready', False)} | Vectors: {graphrag_status.get('vectors_ready', False)}")
                if graphrag_emb_provider and graphrag_emb_model:
                    st.caption(f"Embeddings: {graphrag_emb_provider}/{graphrag_emb_model}")
            else:
                st.info("⚪ GraphRAG not configured")

        # Agent Configuration
        with st.sidebar.expander("🤖 Agent Configuration"):
            st.markdown("Configure the multi-stage agentic RAG system")
            
            enable_agent_web_search = st.checkbox(
                "Enable Web Search (Tavily)", 
                key="agent_web_search",
                help="Enable Tavily web search for Stage 3 retrieval"
            )
            
            st.markdown("**Retrieval Thresholds:**")
            vector_threshold = st.slider("Vector Store Threshold", 0.0, 1.0, 0.6, 0.1, key="vector_thresh")
            dataset_threshold = st.slider("Dataset Threshold", 0.0, 1.0, 0.7, 0.1, key="dataset_thresh")
            
            col_agent_build, col_agent_reset = st.columns(2)
            
            if col_agent_build.button("🔧 Build Agent", use_container_width=True, key="agent_build_btn"):
                if not st.session_state.get("rag_system"):
                    st.error("Please build RAG system first (above)")
                elif not st.session_state.get("enable_emb_sem", False):
                    st.error("Please enable semantic embeddings first")
                elif not selected_llm_provider or not selected_llm_model:
                    st.error("Please select an LLM provider and model first")
                else:
                    _setup_agent(enable_agent_web_search, vector_threshold, dataset_threshold, 
                               selected_llm_provider, selected_llm_model)
            
            if col_agent_reset.button("🗑 Reset Agent", use_container_width=True, key="agent_reset_btn"):
                _reset_agent()

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
        
        if st.session_state.evaluation_history:
            st.metric("Evaluations Completed", len(st.session_state.evaluation_history))
            
            export_format = st.selectbox("Export Format", ["excel", "json"])
            
            if st.button("📥 Export Results"):
                export_results(st.session_state.evaluation_history, export_format)
        else:
            st.text("No evaluations to export yet")
        
        return selected_llm_provider, selected_llm_model, response_type


def manual_evaluation_interface(selected_llm_provider, selected_llm_model, response_type):
    """Manual evaluation interface"""
    
    st.subheader("🔍 Manual Evaluation")
    
    question = st.text_area(
        "Enter your legal question:",
        placeholder="e.g., What is the principle of subsidiarity in EU law?",
        height=100
    )
    
    # Reference answer handling
    reference_mode = st.radio(
        "Reference Answer Source:",
        ["🤖 Auto-find from database", "✏️ Provide manually"],
        horizontal=True
    )
    
    if reference_mode == "🤖 Auto-find from database":
        if st.button("🚀 Generate & Evaluate with Auto-Reference"):
            if not question.strip():
                st.error("❌ Please provide a question")
            else:
                with st.spinner("🤖 Finding reference and evaluating..."):
                    result = st.session_state.evaluator.manual_evaluation_with_lookup(
                        selected_llm_provider, selected_llm_model, question, response_type
                    )
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                        if "suggestion" in result:
                            st.info(result["suggestion"])
                    else:
                        st.session_state.evaluation_history.append(result)
                        st.success("✅ Evaluation completed!")
                        show_evaluation_results(result)
    else:
        reference_answer = st.text_area(
            "Manual Reference Answer:",
            placeholder="Provide the expected/correct answer for evaluation",
            height=150
        )
        
        if st.button("🚀 Generate & Evaluate with Manual Reference"):
            if not question.strip():
                st.error("❌ Please provide a question")
            elif not reference_answer.strip():
                st.error("❌ Please provide a reference answer")
            else:
                with st.spinner("🤖 Generating and evaluating..."):
                    result = st.session_state.evaluator.manual_evaluation(
                        selected_llm_provider, selected_llm_model, question, reference_answer, response_type
                    )
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.session_state.evaluation_history.append(result)
                        st.success("✅ Evaluation completed!")
                        show_evaluation_results(result)

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


def batch_evaluation_interface(selected_llm_provider, selected_llm_model, response_type):
    """Batch evaluation interface - simplified and cleaned"""
    
    st.subheader("🚀 Batch Evaluation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Evaluate multiple questions automatically using database questions and answers")
        
        max_questions = st.number_input(
            "Maximum questions to evaluate:",
            min_value=1,
            max_value=100,
            value=10,
            help="Limit the number of questions for testing"
        )
        
        if st.button("🚀 Start Batch Evaluation", type="primary"):
            with st.spinner("Running batch evaluation..."):
                # Call the corrected batch evaluation method
                results = st.session_state.evaluator.batch_evaluation(
                    selected_llm_provider, selected_llm_model, max_questions, response_type
                )
                
                if results:
                    # Store results in session state
                    st.session_state.evaluation_history.extend(results)
                    
                    # Calculate aggregate scores
                    aggregate_scores = st.session_state.evaluator.calculate_aggregate_scores(results)
                    
                    st.success(f"✅ Batch evaluation completed! Processed {len(results)} questions")
                    
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
                    
                    # Export section
                    st.subheader("📥 Export Results")
                    col_export1, col_export2 = st.columns(2)
                    
                    with col_export1:
                        if st.button("Download Excel Report"):
                            export_results(results, "excel")
                    
                    with col_export2:
                        if st.button("Download JSON Report"):
                            export_results(results, "json")
                    
                    # Detailed analysis in expander
                    with st.expander("📊 Detailed Analysis", expanded=False):
                        display_detailed_analysis(results, selected_llm_provider, selected_llm_model, response_type)
                
                else:
                    st.error("❌ Batch evaluation failed - no results generated")
    
    with col2:
        display_batch_sidebar_info(selected_llm_provider, selected_llm_model)



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
    st.write(f"**Evaluation Method:** Direct database matching (no embeddings)")


def display_batch_sidebar_info(selected_llm_provider, selected_llm_model):
    """Display sidebar info for batch evaluation"""
    st.subheader("⚙️ Batch Settings")
    
    # Statistics about available questions
    if st.session_state.question_bank_ready:
        total_questions = len(st.session_state.evaluator.question_bank.questions)
        st.metric("Available Questions", total_questions)
        st.info("🟢 Database Ready")
    
    # Current session results
    if st.session_state.evaluation_history:
        st.markdown("---")
        st.subheader("📊 Session Results")
        
        # Filter current model results
        current_results = [r for r in st.session_state.evaluation_history 
                          if r.get("llm_provider") == selected_llm_provider 
                          and r.get("llm_model") == selected_llm_model]
        
        if current_results:
            st.write(f"**Current Model:** {len(current_results)} evaluations")
            
            # Quick stats
            avg_scores = {
            'bleu': np.mean([r['evaluation']['bleu_score'] for r in current_results]),
            'rouge': np.mean([r['evaluation']['rouge_score'] for r in current_results]),
            'string': np.mean([r['evaluation']['string_similarity_score'] for r in current_results]),
        }
        if any('semantic_similarity_score' in r['evaluation'] for r in current_results):
            avg_scores['semantic'] = np.mean([r['evaluation'].get('semantic_similarity_score', 0.0) for r in current_results])
            
            st.write(f"Avg BLEU: {avg_scores['bleu']:.3f}")
            st.write(f"Avg ROUGE: {avg_scores['rouge']:.3f}")
            st.write(f"Avg String Sim: {avg_scores['string']:.3f}")
            if 'semantic' in avg_scores:
                st.write(f"Avg Semantic Sim: {avg_scores['semantic']:.3f}")
        
        st.write(f"**Total Session:** {len(st.session_state.evaluation_history)} evaluations")


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
            st.error("Failed to create evaluation LLM")
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
            st.error("Failed to create RAGAS evaluation LLM")
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
        st.warning("No results to export")
        return
    
    # Run LangChain RAG evaluation if enabled
    results = run_langchain_rag_evaluation(results)
    
    # Run RAGAS RAG evaluation if enabled
    results = run_ragas_rag_evaluation(results)
    
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
                    "source_file": result.get("source_file", "")
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
                        "ragas_context_entity_recall_score": ragas_eval.get("context_entity_recall", {}).get("score", ""),
                        "ragas_faithfulness_score": ragas_eval.get("faithfulness", {}).get("score", ""),
                        "ragas_factual_correctness_score": ragas_eval.get("factual_correctness", {}).get("score", ""),
                        "ragas_noise_sensitivity_score": ragas_eval.get("noise_sensitivity", {}).get("score", ""),
                        "ragas_response_relevancy_score": ragas_eval.get("response_relevancy", {}).get("score", ""),
                    })
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            filename = f"llm_evaluation_results_{timestamp}.xlsx"
            
            # Create Excel file in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Evaluation_Results')
            
            # Provide download button
            st.download_button(
                label="📥 Download Excel Report",
                data=output.getvalue(),
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


def _setup_rag(file_obj, persist_dir):
    # 1⃣ stash file to a temp-path so Chroma can reopen it later
    tmp_path = Path(tempfile.gettempdir()) / f"tmp_{file_obj.name}"
    tmp_path.write_bytes(file_obj.getbuffer())

    # 2⃣ reuse the semantic-similarity embedding object already created
    provider  = st.session_state.get("emb_provider_sem")
    modelname = st.session_state.get("emb_model_name_sem")
    if not provider or not modelname:
        st.error("Semantic embedding not initialised.")
        return
    embed_obj = EmbeddingManager.create_embedding_model(provider, modelname)
    if embed_obj is None:
        return

    col_map = {
    "citing": {
        "text" : "TEXT_FROM",
        "celex": "CELEX_FROM",
        "para" : "NUMBER_FROM",
        "title": "TITLE_FROM",
    },
    "cited": {
        "text" : "TEXT_TO",
        "celex": "CELEX_TO",
        "para" : "NUMBER_TO",
        "title": "TITLE_TO",
    },
}
    # 3⃣ spin up / reuse the pipeline
    rag = RAGPipeline(
        dataset_path  = tmp_path,
        persist_dir   = Path(persist_dir),
        embedding     = embed_obj,
        k             = 10,               # Instead of 7 - get more context
        force_rebuild = False,
        col_map       = col_map,
    )
    with st.spinner("Building / loading vector store…"):
        rag.initialise()

    # 4⃣ store handles
    st.session_state.rag_system    = rag
    st.session_state.rag_retriever = rag.retriever
    st.session_state.rag_title_idx = rag.title_index
    st.success("✅ RAG ready!")

def _reset_rag(persist_dir):
    try:
        shutil.rmtree(persist_dir, ignore_errors=True)
    except Exception:
        pass
    for key in ("rag_system", "rag_retriever", "rag_title_idx"):
        st.session_state[key] = None
    st.info("RAG state cleared.")

def display_rag_batch_sidebar_info(selected_llm_provider, selected_llm_model):
    """Display sidebar info for batch RAG Q&A evaluation (no Tips section)"""
    st.subheader("⚙️ Batch Settings (RAG Q&A)")
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
                    "context_entity_recall": [],
                    "faithfulness": [],
                    "factual_correctness": [],
                    "noise_sensitivity": [],
                    "response_relevancy": []
                }
                
                for r in ragas_results:
                    ragas_eval = r["ragas_evaluation"]
                    for metric_name, score_list in ragas_scores.items():
                        if metric_name in ragas_eval and "score" in ragas_eval[metric_name]:
                            score = ragas_eval[metric_name]["score"]
                            if isinstance(score, (int, float)) and not isinstance(score, bool):
                                score_list.append(score)
                
                # Display averages for metrics with data
                metric_display_names = {
                    "context_recall": "Context Recall",
                    "context_precision": "Context Precision", 
                    "context_entity_recall": "Context Entity Recall",
                    "faithfulness": "Faithfulness",
                    "factual_correctness": "Factual Correctness",
                    "noise_sensitivity": "Noise Sensitivity",
                    "response_relevancy": "Response Relevancy"
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

def rag_evaluation_interface(selected_provider, selected_model, response_type):
    st.subheader("🔍 RAG Q&A Evaluation")
    if st.session_state.rag_system is None:
        st.info("Upload dataset and build RAG first.")
        return

    # --- Manual RAG Q&A Evaluation ---
    st.markdown("### Manual RAG Q&A Evaluation")
    question = st.text_area("Enter your legal question for RAG Q&A:", key="rag_manual_question", height=100)
    reference_mode = st.radio(
        "Reference Answer Source (RAG Q&A):",
        ["🤖 Auto-find from database", "✏️ Provide manually"],
        horizontal=True,
        key="rag_reference_mode_manual"
    )
    manual_reference_answer = None
    if reference_mode == "✏️ Provide manually":
        manual_reference_answer = st.text_area(
            "Manual Reference Answer (RAG Q&A):",
            placeholder="Provide the expected/correct answer for evaluation",
            height=150,
            key="rag_manual_reference_answer"
        )
    if st.button("🪄 Generate & Evaluate (RAG Q&A)", key="rag_manual_eval_btn"):
        if not question.strip():
            st.error("❌ Please provide a question")
        elif reference_mode == "✏️ Provide manually" and not manual_reference_answer.strip():
            st.error("❌ Please provide a reference answer")
        else:
            with st.spinner("Generating RAG answer and evaluating..."):
                llm = st.session_state.evaluator.model_manager.create_llm(selected_provider, selected_model)
                from langchain.chains import RetrievalQA
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.rag_retriever,
                    chain_type_kwargs=dict(prompt=_rag_prompt(response_type)),
                    return_source_documents=True
                )
                result = chain.invoke({"query": question})
                answer = result["result"]
                source_docs = result.get("source_documents", [])
                context = "\n\n".join([doc.page_content for doc in source_docs])
                if reference_mode == "🤖 Auto-find from database":
                    reference_info = st.session_state.evaluator.question_bank.find_reference_answer_embedding(question)
                    if reference_info and reference_info.get("question_data"):
                        reference_answer = reference_info["question_data"]["answer_text"]
                    elif reference_info and reference_info.get("answer_text"):
                        reference_answer = reference_info["answer_text"]
                    else:
                        reference_answer = None
                else:
                    reference_answer = manual_reference_answer
                if not reference_answer:
                    st.error("No reference answer found for evaluation.")
                else:
                    evaluation = asyncio.run(
                        st.session_state.evaluator.evaluate_single_response(answer, reference_answer)
                    )
                    rag_result = {
                        "question": question,
                        "generated_answer": answer,
                        "reference_answer": reference_answer,
                        "evaluation": evaluation,
                        "llm_provider": selected_provider,
                        "llm_model": selected_model,
                        "response_type": response_type,
                        "rag": True,
                        "timestamp": datetime.now().isoformat(),
                        "retrieved_context": context,
                        "source_docs": [doc.metadata for doc in source_docs],
                    }
                    
                    # Run LangChain evaluation if enabled
                    if st.session_state.get("enable_rag_eval", False):
                        try:
                            eval_provider = st.session_state.get("eval_llm_provider")
                            eval_model = st.session_state.get("eval_llm_model")
                            if eval_provider and eval_model:
                                eval_llm = ModelManager.create_llm(eval_provider, eval_model)
                                if eval_llm:
                                    eval_pipeline = LangchainEvalPipeline(eval_llm)
                                    lc_eval = eval_pipeline.evaluate_all(
                                        question=question,
                                        reference_answer=reference_answer,
                                        generated_answer=answer,
                                        context=context
                                    )
                                    rag_result["langchain_evaluation"] = lc_eval
                                    st.success("✅ LangChain evaluation completed!")
                        except Exception as e:
                            st.warning(f"LangChain evaluation failed: {str(e)}")
                    
                    # Run RAGAS evaluation if enabled
                    if st.session_state.get("enable_ragas_eval", False):
                        try:
                            ragas_provider = st.session_state.get("ragas_llm_provider")
                            ragas_model = st.session_state.get("ragas_llm_model")
                            if ragas_provider and ragas_model:
                                ragas_llm = ModelManager.create_llm(ragas_provider, ragas_model)
                                if ragas_llm:
                                    # Create embeddings if enabled
                                    ragas_embeddings = None
                                    if st.session_state.get("ragas_use_embeddings", False):
                                        ragas_emb_provider = st.session_state.get("ragas_emb_provider")
                                        ragas_emb_model = st.session_state.get("ragas_emb_model")
                                        if ragas_emb_provider and ragas_emb_model:
                                            ragas_embeddings = EmbeddingManager.create_embedding_model(ragas_emb_provider, ragas_emb_model)
                                    
                                    ragas_pipeline = RAGASEvalPipeline(ragas_llm, ragas_embeddings)
                                    ragas_eval = asyncio.run(ragas_pipeline.evaluate_all(
                                        question=question,
                                        reference_answer=reference_answer,
                                        generated_answer=answer,
                                        context=context
                                    ))
                                    rag_result["ragas_evaluation"] = ragas_eval
                                    st.success("✅ RAGAS evaluation completed!")
                        except Exception as e:
                            st.warning(f"RAGAS evaluation failed: {str(e)}")
                    
                    st.session_state.rag_evaluation_history.append(rag_result)
                    show_rag_evaluation_results(rag_result)
    st.markdown("---")
    # --- Batch RAG Q&A Evaluation ---
    st.markdown("### Batch RAG Q&A Evaluation")
    col1, col2 = st.columns([2, 1])
    with col1:
        max_questions = st.number_input(
            "Maximum questions to evaluate (RAG Q&A):",
            min_value=1,
            max_value=100,
            value=10,
            key="rag_batch_max_questions"
        )
        if st.button("🚀 Start Batch RAG Q&A Evaluation", key="rag_batch_eval_btn"):
            with st.spinner("Running batch RAG Q&A evaluation..."):
                questions = st.session_state.evaluator.question_bank.get_all_questions()
                questions = questions[:max_questions]
                results = []
                llm = st.session_state.evaluator.model_manager.create_llm(selected_provider, selected_model)
                from langchain.chains import RetrievalQA
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.rag_retriever,
                    chain_type_kwargs=dict(prompt=_rag_prompt(response_type)),
                    return_source_documents=True
                )
                progress_bar = st.progress(0)
                for idx, qdata in enumerate(questions):
                    question_text = qdata["question_text"]
                    reference_answer = qdata["answer_text"]
                    try:
                        result = chain.invoke({"query": question_text})
                        answer = result["result"]
                        source_docs = result.get("source_documents", [])
                        context = "\n\n".join([doc.page_content for doc in source_docs])
                        evaluation = asyncio.run(
                            st.session_state.evaluator.evaluate_single_response(answer, reference_answer)
                        )
                        rag_result = {
                            "question_id": qdata.get("id"),
                            "year": qdata.get("year"),
                            "question_number": qdata.get("question_number"),
                            "question": question_text,
                            "generated_answer": answer,
                            "reference_answer": reference_answer,
                            "evaluation": evaluation,
                            "llm_provider": selected_provider,
                            "llm_model": selected_model,
                            "response_type": response_type,
                            "rag": True,
                            "timestamp": datetime.now().isoformat(),
                            "retrieved_context": context,
                            "source_docs": [doc.metadata for doc in source_docs],
                            "source_file": qdata.get("source_file", "")
                        }
                        results.append(rag_result)
                    except Exception as e:
                        st.error(f"Error processing question {idx + 1}: {str(e)}")
                        continue
                    progress_bar.progress((idx + 1) / len(questions))
                progress_bar.empty()
                
                # Run LangChain evaluation on batch results if enabled
                if st.session_state.get("enable_rag_eval", False) and results:
                    try:
                        eval_provider = st.session_state.get("eval_llm_provider")
                        eval_model = st.session_state.get("eval_llm_model")
                        if eval_provider and eval_model:
                            eval_llm = ModelManager.create_llm(eval_provider, eval_model)
                            if eval_llm:
                                eval_pipeline = LangchainEvalPipeline(eval_llm)
                                st.info(f"Running LangChain evaluation on {len(results)} batch results...")
                                
                                # Prepare samples for batch evaluation
                                samples = []
                                for result in results:
                                    samples.append({
                                        "question": result.get("question", ""),
                                        "reference_answer": result.get("reference_answer", ""),
                                        "generated_answer": result.get("generated_answer", ""),
                                        "context": result.get("retrieved_context", "")
                                    })
                                
                                # Run batch evaluation
                                with st.spinner("Running LangChain batch evaluation..."):
                                    eval_results = eval_pipeline.batch_evaluate(samples)
                                
                                # Add LangChain evaluation to results
                                for i, result in enumerate(results):
                                    if i < len(eval_results):
                                        result["langchain_evaluation"] = eval_results[i]["evaluation"]
                                
                                st.success(f"✅ LangChain evaluation completed for {len(results)} batch results!")
                    except Exception as e:
                        st.warning(f"LangChain evaluation failed: {str(e)}")
                
                st.session_state.rag_evaluation_history.extend(results)
                if results:
                    st.success(f"✅ Batch RAG Q&A evaluation completed! Processed {len(results)} questions")
                    # Results Summary
                    aggregate_scores = st.session_state.evaluator.calculate_aggregate_scores(results)
                    st.subheader("📊 Results Summary (RAG Q&A)")
                    col_a, col_b, col_c, col_d, col_e = st.columns(5)
                    with col_a:
                        st.metric("Total Questions", len(results))
                    with col_b:
                        st.metric("Avg BLEU", f"{aggregate_scores.get('Avg BLEU', 0):.3f}")
                    with col_c:
                        st.metric("Avg ROUGE", f"{aggregate_scores.get('Avg ROUGE', 0):.3f}")
                    with col_d:
                        st.metric("Avg String Similarity", f"{aggregate_scores.get('Avg String Similarity', 0):.3f}")
                    with col_e:
                        if 'Avg Semantic Similarity' in aggregate_scores:
                            st.metric("Avg Semantic Similarity", f"{aggregate_scores['Avg Semantic Similarity']:.3f}")
                    
                    # LangChain metrics summary if available
                    lc_results = [r for r in results if "langchain_evaluation" in r]
                    if lc_results:
                        st.subheader("🔬 LangChain Metrics Summary")
                        lc_col1, lc_col2, lc_col3, lc_col4 = st.columns(4)
                        
                        # Calculate LangChain averages
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
                        
                        with lc_col1:
                            if lc_correctness_scores:
                                avg_correctness = np.mean(lc_correctness_scores)
                                color = "🟢" if avg_correctness >= 4 else "🟡" if avg_correctness >= 3 else "🔴"
                                st.metric("Avg Correctness", f"{avg_correctness:.1f}/5")
                                st.caption(f"{color} {_get_score_description(int(avg_correctness), 'correctness')}")
                        
                        with lc_col2:
                            if lc_relevance_scores:
                                avg_relevance = np.mean(lc_relevance_scores)
                                color = "🟢" if avg_relevance >= 4 else "🟡" if avg_relevance >= 3 else "🔴"
                                st.metric("Avg Relevance", f"{avg_relevance:.1f}/5")
                                st.caption(f"{color} {_get_score_description(int(avg_relevance), 'relevance')}")
                        
                        with lc_col3:
                            if lc_groundedness_scores:
                                avg_groundedness = np.mean(lc_groundedness_scores)
                                color = "🟢" if avg_groundedness >= 4 else "🟡" if avg_groundedness >= 3 else "🔴"
                                st.metric("Avg Groundedness", f"{avg_groundedness:.1f}/5")
                                st.caption(f"{color} {_get_score_description(int(avg_groundedness), 'groundedness')}")
                        
                        with lc_col4:
                            if lc_retrieval_scores:
                                avg_retrieval = np.mean(lc_retrieval_scores)
                                color = "🟢" if avg_retrieval >= 4 else "🟡" if avg_retrieval >= 3 else "🔴"
                                st.metric("Avg Retrieval Relevance", f"{avg_retrieval:.1f}/5")
                                st.caption(f"{color} {_get_score_description(int(avg_retrieval), 'retrieval_relevance')}")
                        
                        st.caption(f"LangChain evaluated: {len(lc_results)}/{len(results)} results")
                    # Results table preview
                    st.subheader("📋 Results Preview (RAG Q&A)")
                    display_results_preview(results)
                    # Export section (batch only)
                    st.subheader("📥 Export Results (RAG Q&A)")
                    col_export1, col_export2, col_export3 = st.columns(3)
                    with col_export1:
                        if st.button("Download Excel Report (RAG Q&A)", key="rag_excel_export_btn"):
                            export_results(results, "excel")
                    with col_export2:
                        if st.button("Download JSON Report (RAG Q&A)", key="rag_json_export_btn"):
                            export_results(results, "json")
                    with col_export3:
                        if st.button("🔬 Run LangChain Eval", key="rag_langchain_eval_btn"):
                            if st.session_state.get("enable_rag_eval", False):
                                enhanced_results = run_langchain_rag_evaluation(results)
                                st.session_state.rag_evaluation_history = enhanced_results
                                st.success("✅ LangChain evaluation completed!")
                                st.rerun()
                            else:
                                st.warning("Please enable LangChain RAG Evaluation in the sidebar first")
                    # Detailed analysis in expander
                    with st.expander("📊 Detailed Analysis (RAG Q&A)", expanded=False):
                        display_detailed_analysis(results, selected_provider, selected_model, response_type)
                else:
                    st.error("❌ Batch RAG Q&A evaluation failed - no results generated")
    with col2:
        display_rag_batch_sidebar_info(selected_provider, selected_model)
        # Sidebar export for full session
        st.markdown("---")
        st.subheader("📊 Export Session Results (RAG Q&A)")
        if st.session_state.rag_evaluation_history:
            st.metric("Evaluations Completed", len(st.session_state.rag_evaluation_history))
            export_format = st.selectbox("Export Format (RAG Q&A)", ["excel", "json"], key="rag_export_format")
            if st.button("📥 Export Session Results (RAG Q&A)", key="rag_export_btn"):
                export_results(st.session_state.rag_evaluation_history, export_format)
        else:
            st.text("No RAG Q&A evaluations to export yet")

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
        
        # Show RAGAS system status in a small expander
        with st.expander("🔧 RAGAS System Status", expanded=False):
            # Try to get system status if the pipeline is available in session state
            ragas_provider = st.session_state.get("ragas_llm_provider")
            ragas_model = st.session_state.get("ragas_llm_model")
            
            if ragas_provider and ragas_model:
                st.markdown(f"**Evaluation LLM:** {ragas_provider}/{ragas_model}")
            
            ragas_emb_provider = st.session_state.get("ragas_emb_provider")
            ragas_emb_model = st.session_state.get("ragas_emb_model")
            
            if ragas_emb_provider and ragas_emb_model:
                st.markdown(f"**Embeddings Model:** {ragas_emb_provider}/{ragas_emb_model}")
            elif st.session_state.get("ragas_use_embeddings", False):
                st.warning("Embeddings requested but not configured")
            else:
                st.info("Response Relevancy disabled (no embeddings)")
            
            # Show which metrics had issues
            ragas_eval = result["ragas_evaluation"]
            failed_metrics = []
            successful_metrics = []
            
            for metric_name in ["context_recall", "context_precision", "context_entity_recall", 
                              "faithfulness", "factual_correctness", "noise_sensitivity", "response_relevancy"]:
                if metric_name in ragas_eval:
                    if "error" in ragas_eval[metric_name] or ragas_eval[metric_name].get("score", 0) == 0:
                        failed_metrics.append(metric_name)
                    else:
                        successful_metrics.append(metric_name)
            
            col_status1, col_status2 = st.columns(2)
            with col_status1:
                st.metric("Successful Metrics", len(successful_metrics))
                if successful_metrics:
                    st.caption(", ".join(successful_metrics))
            
            with col_status2:
                st.metric("Failed/Zero Metrics", len(failed_metrics))
                if failed_metrics:
                    st.caption(", ".join(failed_metrics))

def show_langchain_evaluation_results(lc_eval: Dict[str, Any]):
    """Display LangChain RAG evaluation results"""
    st.markdown("---")
    st.subheader("🔬 LangChain RAG Evaluation Results")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        correctness = lc_eval.get("correctness", {})
        score = correctness.get("score", 0)
        color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
        st.metric("Correctness", f"{score}/5")
        st.caption(f"{color} {_get_score_description(score, 'correctness')}")
    
    with col2:
        relevance = lc_eval.get("relevance", {})
        score = relevance.get("score", 0)
        color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
        st.metric("Relevance", f"{score}/5")
        st.caption(f"{color} {_get_score_description(score, 'relevance')}")
    
    with col3:
        groundedness = lc_eval.get("groundedness", {})
        score = groundedness.get("score", 0)
        color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
        st.metric("Groundedness", f"{score}/5")
        st.caption(f"{color} {_get_score_description(score, 'groundedness')}")
    
    with col4:
        retrieval_rel = lc_eval.get("retrieval_relevance", {})
        score = retrieval_rel.get("score", 0)
        color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
        st.metric("Retrieval Relevance", f"{score}/5")
        st.caption(f"{color} {_get_score_description(score, 'retrieval_relevance')}")
    
    # Detailed explanations in expander (for debugging/validation purposes)
    with st.expander("📝 Detailed Explanations (Debug)", expanded=False):
        st.caption("💡 These explanations are used internally for better evaluation quality but not exported for zero-shot Q&A systems")
        
        st.markdown("**Correctness Explanation:**")
        st.write(correctness.get("explanation", "No explanation available"))
        
        st.markdown("**Relevance Explanation:**")
        st.write(relevance.get("explanation", "No explanation available"))
        
        st.markdown("**Groundedness Explanation:**")
        st.write(groundedness.get("explanation", "No explanation available"))
        
        st.markdown("**Retrieval Relevance Explanation:**")
        st.write(retrieval_rel.get("explanation", "No explanation available"))

def show_ragas_evaluation_results(ragas_eval: Dict[str, Any]):
    """Display RAGAS RAG evaluation results with all 7 metrics"""
    st.markdown("---")
    st.subheader("📊 RAGAS RAG Evaluation Results")
    
    # Define all metrics with their display information
    metrics_info = [
        ("context_recall", "Context Recall", "🔍"),
        ("context_precision", "Context Precision", "🎯"),
        ("context_entity_recall", "Entity Recall", "🏷️"),
        ("faithfulness", "Faithfulness", "✅"),
        ("factual_correctness", "Factual Correctness", "📋"),
        ("noise_sensitivity", "Noise Sensitivity", "🛡️"),
        ("response_relevancy", "Response Relevancy", "💭")
    ]
    
    # Create two rows of metrics (4 + 3)
    col1, col2, col3, col4 = st.columns(4)
    cols_row1 = [col1, col2, col3, col4]
    
    # First row - 4 metrics
    for i, (metric_key, metric_name, emoji) in enumerate(metrics_info[:4]):
        with cols_row1[i]:
            metric_data = ragas_eval.get(metric_key, {})
            score = metric_data.get("score", 0.0)
            
            if "error" in metric_data:
                st.metric(f"{emoji} {metric_name}", "Error")
                st.caption("🔴 Evaluation failed")
            else:
                color = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
                st.metric(f"{emoji} {metric_name}", f"{score:.3f}")
                st.caption(f"{color} {_get_ragas_score_description(score, metric_key)}")
    
    # Second row - 3 metrics
    col5, col6, col7 = st.columns(3)
    cols_row2 = [col5, col6, col7]
    
    for i, (metric_key, metric_name, emoji) in enumerate(metrics_info[4:]):
        with cols_row2[i]:
            metric_data = ragas_eval.get(metric_key, {})
            score = metric_data.get("score", 0.0)
            
            if "error" in metric_data:
                st.metric(f"{emoji} {metric_name}", "Error")
                st.caption("🔴 Evaluation failed")
            elif metric_key == "response_relevancy" and score == 0.0:
                st.metric(f"{emoji} {metric_name}", "N/A")
                st.caption("⚫ Requires embeddings")
            else:
                color = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
                st.metric(f"{emoji} {metric_name}", f"{score:.3f}")
                st.caption(f"{color} {_get_ragas_score_description(score, metric_key)}")
    
    # Metric descriptions in expander
    with st.expander("📚 RAGAS Metric Descriptions", expanded=False):
        st.markdown("**Context Recall:** Measures how much of the relevant information from the reference is captured in the retrieved context. Higher values indicate better retrieval completeness.")
        st.markdown("**Context Precision:** Measures the precision of retrieved context by evaluating what proportion of the retrieved context is relevant to answering the question.")
        st.markdown("**Context Entity Recall:** Measures recall based on entities present in ground truth and context. Useful for entity-focused legal applications.")
        st.markdown("**Faithfulness:** Measures the factual consistency of the generated answer against the retrieved context. Higher values indicate less hallucination.")
        st.markdown("**Factual Correctness:** Measures the factual accuracy of the generated answer compared to the reference answer using high atomicity and coverage for detailed legal analysis.")
        st.markdown("**Noise Sensitivity:** Measures how robust the system is to irrelevant or noisy information in the retrieved context.")
        st.markdown("**Response Relevancy:** Measures how relevant and focused the generated answer is to the input question. Higher values indicate more relevant responses.")
        
        # Show FactualCorrectness configuration
        st.markdown("---")
        st.markdown("**FactualCorrectness Configuration:**")
        st.markdown("• Mode: F1 (balanced precision and recall)")
        st.markdown("• Atomicity: High (detailed claim decomposition)")
        st.markdown("• Coverage: High (comprehensive evaluation)")
        st.markdown("• Optimized for legal Q&A systems")
        
        # Show any errors
        for metric_key, metric_name, _ in metrics_info:
            if metric_key in ragas_eval and "error" in ragas_eval[metric_key]:
                st.error(f"**{metric_name} Error:** {ragas_eval[metric_key]['error']}")
    
    # Debug expander for detailed LLM reasoning
    with st.expander("🔧 RAGAS Debug Information", expanded=False):
        st.markdown("**Enhanced Debug Information with Error Analysis**")
        st.caption("⚠️ This section shows detailed LLM reasoning, error analysis, and troubleshooting information")
        
        # Error summary first
        metrics_with_errors = []
        parsing_errors = []
        timeout_errors = []
        zero_scores = []
        
        for metric_key, metric_name, emoji in metrics_info:
            if metric_key in ragas_eval:
                metric_data = ragas_eval[metric_key]
                score = metric_data.get("score", 0.0)
                
                if "error" in metric_data:
                    metrics_with_errors.append(f"{emoji} {metric_name}")
                    error_category = metric_data.get("error_category", "Unknown")
                    if "Parsing" in error_category:
                        parsing_errors.append(metric_name)
                    elif "Timeout" in error_category:
                        timeout_errors.append(metric_name)
                elif score == 0.0:
                    zero_scores.append(f"{emoji} {metric_name}")
        
        # Error Summary Section
        st.markdown("**🚨 Error Summary:**")
        col_err1, col_err2, col_err3, col_err4 = st.columns(4)
        
        with col_err1:
            st.metric("Metrics with Errors", len(metrics_with_errors))
            if metrics_with_errors:
                st.caption("\n".join(metrics_with_errors))
        
        with col_err2:
            st.metric("Parsing Errors", len(parsing_errors))
            if parsing_errors:
                st.caption("JSON/Schema issues")
        
        with col_err3:
            st.metric("Timeout Errors", len(timeout_errors))
            if timeout_errors:
                st.caption("Evaluation too slow")
        
        with col_err4:
            st.metric("Zero Scores", len(zero_scores))
            if zero_scores:
                st.caption("No valid evaluation")
        
        st.markdown("---")
        
        # Detailed metric analysis
        for metric_key, metric_name, emoji in metrics_info:
            if metric_key in ragas_eval:
                metric_data = ragas_eval[metric_key]
                
                st.markdown(f"**{emoji} {metric_name} Analysis:**")
                
                # Score and status
                score = metric_data.get("score", 0.0)
                description = metric_data.get("description", "No description available")
                
                col_metric1, col_metric2 = st.columns([1, 2])
                with col_metric1:
                    st.write(f"**Score:** {score}")
                    if "evaluation_successful" in metric_data:
                        status = "✅ Success" if metric_data["evaluation_successful"] else "❌ Failed"
                        st.write(f"**Status:** {status}")
                
                with col_metric2:
                    st.write(f"**Description:** {description}")
                
                # Error analysis if present
                if "error" in metric_data:
                    st.error(f"**Error:** {metric_data['error']}")
                    
                    if "error_category" in metric_data:
                        st.warning(f"**Category:** {metric_data['error_category']}")
                    
                    if "suggestion" in metric_data:
                        st.info(f"**Suggestion:** {metric_data['suggestion']}")
                
                # LLM Reasoning (new enhanced section)
                if "llm_reasoning" in metric_data:
                    st.markdown(f"**🧠 LLM Reasoning - {metric_name}:**")
                    llm_reasoning = metric_data["llm_reasoning"]
                    if isinstance(llm_reasoning, str) and len(llm_reasoning) > 2000:
                        st.code(f"{llm_reasoning[:1000]}\n\n... [TRUNCATED - {len(llm_reasoning)} total chars] ...\n\n{llm_reasoning[-1000:]}")
                    else:
                        st.code(str(llm_reasoning))
                
                # Input Analysis (new section to understand zero scores)
                st.markdown(f"**📊 Input Analysis - {metric_name}:**")
                if "input_question" in metric_data:
                    st.write(f"• **Question:** {metric_data['input_question']}")
                if "input_context_length" in metric_data:
                    st.write(f"• **Context Length:** {metric_data['input_context_length']} characters")
                if "input_reference_length" in metric_data:
                    st.write(f"• **Reference Length:** {metric_data['input_reference_length']} characters")
                if "input_response_length" in metric_data:
                    st.write(f"• **Generated Response Length:** {metric_data['input_response_length']} characters")
                
                # Zero Score Analysis (new section to explain why scores are zero)
                score = metric_data.get("score", 0.0)
                if score == 0.0 and "error" not in metric_data:
                    st.markdown(f"**🔍 Zero Score Analysis - {metric_name}:**")
                    if metric_name in ["Context Recall", "Context Precision", "Context Entity Recall"]:
                        st.info(f"**Why {metric_name} is 0.0:** This is normal for legal Q&A systems where retrieved context is focused/specific but reference answers are comprehensive. The context may not contain all the information in the reference answer, leading to low recall/precision scores.")
                    elif metric_name == "Noise Sensitivity":
                        st.info(f"**Why {metric_name} is 0.0:** This metric often has compatibility issues with certain LLM models due to JSON parsing requirements. The LLM may not be producing output in the exact format expected by RAGAS.")
                    else:
                        st.info(f"**Why {metric_name} is 0.0:** The evaluation completed without errors but returned a zero score. This could indicate a mismatch between the evaluation criteria and your specific use case.")
                
                # Raw response (fallback if no llm_reasoning)
                if "raw_response" in metric_data and "llm_reasoning" not in metric_data:
                    st.markdown(f"**Raw Response - {metric_name}:**")
                    raw_response = metric_data["raw_response"]
                    if isinstance(raw_response, str) and len(raw_response) > 1000:
                        st.code(f"{raw_response[:500]}\n\n... [TRUNCATED - {len(raw_response)} total chars] ...\n\n{raw_response[-500:]}")
                    else:
                        st.code(str(raw_response))
                
                # Additional debug info (show remaining keys)
                debug_keys = [k for k in metric_data.keys() if k not in ["score", "description", "raw_response", "llm_reasoning", "input_question", "input_context_length", "input_reference_length", "input_response_length", "error", "error_category", "suggestion", "evaluation_successful"]]
                if debug_keys:
                    st.markdown(f"**Additional Info - {metric_name}:**")
                    for key in debug_keys:
                        st.write(f"• **{key.replace('_', ' ').title()}:** {metric_data[key]}")
                
                st.markdown("---")
        
        # Overall evaluation context
        st.markdown("**📊 Evaluation Context Summary:**")
        total_metrics = len(metrics_info)
        evaluated_metrics = len([m for m in metrics_info if m[0] in ragas_eval])
        successful_metrics = len([m for m in metrics_info if m[0] in ragas_eval and ragas_eval[m[0]].get("score", 0) > 0])
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        with col_sum1:
            st.metric("Total Metrics", total_metrics)
        with col_sum2:
            st.metric("Evaluated", evaluated_metrics)
        with col_sum3:
            st.metric("Successful", successful_metrics)
        
        # Root cause analysis
        st.markdown("**🔍 Root Cause Analysis:**")
        
        if parsing_errors:
            st.error(f"**Parsing Issues ({len(parsing_errors)} metrics):** The evaluation LLM is not producing output in the format expected by RAGAS. This is common with certain models (like Gemini) that may format JSON differently.")
            st.markdown("**Solutions:** Try a different evaluation LLM (e.g., GPT-4, Claude) or adjust RAGAS prompt templates.")
        
        if timeout_errors:
            st.warning(f"**Timeout Issues ({len(timeout_errors)} metrics):** Some evaluations are taking too long, possibly due to complex prompts or slow model responses.")
            st.markdown("**Solutions:** Use faster models, reduce context length, or increase timeout values.")
        
        if len(zero_scores) > len(metrics_with_errors):
            st.info(f"**Zero Scores ({len(zero_scores) - len(metrics_with_errors)} metrics):** These metrics completed without errors but returned 0.0 scores, often due to context-reference mismatches in legal Q&A systems.")
            st.markdown("**Note:** This is expected when retrieved context doesn't contain all reference answer information.")
        
        # Troubleshooting guide
        st.markdown("**🛠️ Troubleshooting Guide:**")
        troubleshooting_tips = [
            "**Model Compatibility:** Try GPT-4 or Claude instead of Gemini for better RAGAS compatibility",
            "**Context Quality:** Ensure retrieved context is relevant and properly formatted",
            "**Reference Answers:** Use concise, focused reference answers rather than comprehensive ones",
            "**Timeout Issues:** Reduce context length or use faster evaluation models",
            "**Parsing Errors:** Check if your LLM follows JSON schema requirements strictly",
            "**Zero Scores:** Normal for legal Q&A where context may not contain all reference information"
        ]
        
        for tip in troubleshooting_tips:
            st.markdown(f"• {tip}")
        
        # Model recommendations
        st.markdown("**🎯 Model Recommendations for RAGAS:**")
        st.markdown("• **Best:** OpenAI GPT-4 (highest compatibility)")
        st.markdown("• **Good:** Anthropic Claude (good JSON adherence)")
        st.markdown("• **Fair:** Google Gemini (may have parsing issues)")
        st.markdown("• **Avoid:** Smaller models (insufficient reasoning capability)")
        
        # Detailed metric explanations
        st.markdown("---")
        st.markdown("**📚 Why Context Metrics Show Zero Scores:**")
        
        st.markdown("**Context Recall (0.0):** Measures what percentage of information from the reference answer is found in the retrieved context. In legal Q&A:")
        st.markdown("• Reference answers are comprehensive and cover broad legal principles")
        st.markdown("• Retrieved context is focused on specific cases/paragraphs")
        st.markdown("• **Zero score is normal** - context doesn't contain all reference information")
        
        st.markdown("**Context Precision (0.0):** Measures what percentage of retrieved context is relevant to answering the question. Zero scores indicate:")
        st.markdown("• The LLM evaluator doesn't see the context as directly useful for the specific question")
        st.markdown("• Context may be legally relevant but not explicitly mentioned in reference")
        st.markdown("• **This is expected** when context provides supporting background rather than direct answers")
        
        st.markdown("**Entity Recall (0.0):** Measures overlap of legal entities (cases, laws, articles) between context and reference. Zero scores mean:")
        st.markdown("• Reference answer mentions different cases/laws than those in retrieved context")
        st.markdown("• Context provides relevant legal principles but different specific citations")
        st.markdown("• **Normal for legal Q&A** where multiple valid authorities exist")
        
        st.markdown("**Noise Sensitivity:** Measures robustness to irrelevant information. Input/Output format:")
        st.markdown("• **Input:** Question + Answer + Context (with some irrelevant parts)")
        st.markdown("• **Expected Output:** JSON with robustness score (0.0-1.0)")
        st.markdown("• **Common Issues:** LLM doesn't follow exact JSON schema, parsing failures")
        st.markdown("• **Why it fails:** Strict JSON requirements that vary by model")
        
        st.markdown("**💡 Focus on Response Relevancy and Faithfulness** - these are the most reliable indicators for legal Q&A quality!")

def _get_ragas_score_description(score: float, metric_type: str) -> str:
    """Get description for RAGAS score (0.0-1.0 scale) for all 7 metrics"""
    if score >= 0.8:
        descriptions = {
            "context_recall": "Excellent Retrieval",
            "context_precision": "Highly Precise",
            "context_entity_recall": "Excellent Entity Coverage",
            "faithfulness": "Highly Faithful",
            "factual_correctness": "Highly Accurate",
            "noise_sensitivity": "Highly Robust",
            "response_relevancy": "Highly Relevant"
        }
        return descriptions.get(metric_type, "Excellent")
    elif score >= 0.6:
        descriptions = {
            "context_recall": "Good Retrieval",
            "context_precision": "Precise",
            "context_entity_recall": "Good Entity Coverage",
            "faithfulness": "Faithful",
            "factual_correctness": "Accurate",
            "noise_sensitivity": "Robust",
            "response_relevancy": "Relevant"
        }
        return descriptions.get(metric_type, "Good")
    elif score >= 0.4:
        descriptions = {
            "context_recall": "Partial Retrieval",
            "context_precision": "Somewhat Precise",
            "context_entity_recall": "Partial Entity Coverage",
            "faithfulness": "Somewhat Faithful",
            "factual_correctness": "Partially Accurate",
            "noise_sensitivity": "Somewhat Robust",
            "response_relevancy": "Somewhat Relevant"
        }
        return descriptions.get(metric_type, "Fair")
    else:
        descriptions = {
            "context_recall": "Poor Retrieval",
            "context_precision": "Imprecise",
            "context_entity_recall": "Poor Entity Coverage",
            "faithfulness": "Unfaithful",
            "factual_correctness": "Inaccurate",
            "noise_sensitivity": "Not Robust",
            "response_relevancy": "Not Relevant"
        }
        return descriptions.get(metric_type, "Poor")

def _get_score_description(score: int, metric_type: str) -> str:
    """Get description for score"""
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

def _rag_prompt(style):
    """
    Hybrid-RAG prompt   ·   v2-June-2025
    ─────────────────────────────────────
    • Grounds where possible, but lets the LLM draw on its parametric EU-law knowledge
      when the retrieved context is silent.
    • Uses quotation-marks around every cited case title so downstream post-processing
      can extract them reliably.
    • Adds a negative example to discourage hallucinated / invented case names.
    • For the "concise" style, hard-caps the answer at ~120 words.
    """

    common_rules = (
    "INSTRUCTIONS:\n"
    "• Provide a complete, accurate answer using your EU-law expertise.\n"
    "• When a sentence is directly supported or illustrated by a passage in <context>, "
    "append the case title **exactly as it appears in <context>, inside quotation marks** "
    "and include a short quote or paraphrase (e.g. \"Costa v ENEL\").\n"
    "• Do **not** invent or guess case names – cite only those that occur in <context>.\n"
    "• If the context is silent on a point, rely on general EU jurisprudence, but **do not "
    "mention that the context was missing**.\n"
    "• Write in clear, well-structured paragraphs (no bullet points).\n"
    )

    # ── 2. Negative example (hallucinated citation) ────────────────────────────────
    negative_example = (
        "NEGATIVE EXAMPLE (hallucinated citation):\n"
        "Context: [Commission v Greece] The Greek government failed to implement Directive 91/271/EEC...\n\n"
        "Question: What are the consequences of failing to implement EU directives?\n\n"
        "❌ Bad answer (do NOT copy this): Member States may also face sanctions established in "
        "\"Fictional v MemberState\". ← This case is NOT in <context>, so the citation is invalid.\n\n"
    )

    # ── 3. Positive example (proper citation usage) ────────────────────────────────
    positive_example = (
        "POSITIVE EXAMPLE:\n"
        "Context:\n"
        "[Commission v Greece] The Greek government failed to implement Directive 91/271/EEC "
        "within the prescribed timeframe...\n\n"
        "Question: What are the consequences of failing to implement EU directives?\n\n"
        "✅ Good answer: Member States face several consequences when they fail to implement directives. "
        "The Commission may initiate infringement proceedings under Article 258 TFEU. As shown in "
        "\"Commission v Greece\", legal action can follow when a directive is not implemented "
        "\"within the prescribed timeframe\".  In addition, the Francovich doctrine establishes that "
        "Member States are liable in damages for failure to transpose directives, creating enforceable "
        "rights for individuals—though this particular case is not detailed in the current context.\n\n"
    )


    header = (
        "You are an EU-law specialist. Answer the following question as instructed.\n\n"
    )

    # ──────────────────────────────────────────────────────────────────────────
    if style.lower() == "detailed":
        tmpl = (
            header
            + common_rules
            + "• Aim for a comprehensive answer (several paragraphs) covering principles, exceptions, and rationale.\n"
            "\n"
            + negative_example
            + positive_example
            + "<context>\n{context}\n</context>\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    else:  # concise
        tmpl = (
            header
            + common_rules
            + "• Provide a single paragraph **no longer than 120 words**.\n"
            "\n"
            + negative_example
            + positive_example
            + "<context>\n{context}\n</context>\n\n"
            "Question: {question}\n\n"
            "Answer (≤ 250 words):"
        )
    
    from langchain.prompts import PromptTemplate
    return PromptTemplate(
        input_variables=["context", "question"],
        template=tmpl,
    )

# Additional debugging function you can add
def debug_retriever():
    """Debug function to test if retriever is working"""
    if st.session_state.rag_retriever is None:
        st.error("No retriever found")
        return
    
    test_query = "test query"
    try:
        docs = st.session_state.rag_retriever.get_relevant_documents(test_query)
        st.write(f"Retriever test: Found {len(docs)} documents")
        if docs:
            st.write("First doc content:", docs[0].page_content[:200])
            st.write("First doc metadata:", docs[0].metadata)
    except Exception as e:
        st.error(f"Retriever error: {e}")

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

def _setup_agent(enable_web_search, vector_thresh, dataset_thresh, selected_provider=None, selected_model=None):
    """Setup the multi-stage agent system"""
    try:
        with st.spinner("Building agent system..."):
            # Use a fixed path for the main dataset or get it from session state
            main_dataset_path = "./PAR-TO-PAR (MAIN).csv"
            
            if not Path(main_dataset_path).exists():
                st.error(f"Main dataset not found at {main_dataset_path}")
                st.info("Please ensure your main dataset (110k rows) is available at the specified path")
                return
            
            # Load main dataset with proper encoding handling
            try:
                # Try UTF-8 first
                main_df = pd.read_csv(main_dataset_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    # Try latin-1 (common for European legal documents)
                    main_df = pd.read_csv(main_dataset_path, encoding='latin-1')
                    st.info("Loaded dataset using latin-1 encoding")
                except UnicodeDecodeError:
                    try:
                        # Try cp1252 (Windows encoding)
                        main_df = pd.read_csv(main_dataset_path, encoding='cp1252')
                        st.info("Loaded dataset using cp1252 encoding")
                    except UnicodeDecodeError:
                        # Last resort: ignore errors
                        main_df = pd.read_csv(main_dataset_path, encoding='utf-8', errors='ignore')
                        st.warning("Loaded dataset with some character encoding issues ignored")
            
            st.write(f"Loaded main dataset: {len(main_df)} rows")
            st.write(f"Dataset columns: {list(main_df.columns)}")
            
            # Use the selected LLM from sidebar
            providers = ModelManager.get_available_llm_providers()
            if not providers:
                st.error("No LLM providers available")
                return
            
            # Use the passed provider and model, or fall back to first available
            if selected_provider and selected_model:
                provider = selected_provider
                model = selected_model
                st.info(f"Using selected LLM: {provider}/{model}")
            else:
                # Fallback to first available
                provider = list(providers.keys())[0]
                model = providers[provider]["models"][0]
                st.warning(f"No LLM selected, using fallback: {provider}/{model}")
            
            llm = ModelManager.create_llm(provider, model)
            
            if not llm:
                st.error("Failed to create LLM")
                return
            
            st.success(f"✅ LLM created: {provider}/{model}")
            
            # Setup web search if enabled
            web_search_tool = None
            if enable_web_search:
                try:
                    web_search_tool = WebSearchTool()  # Will use TAVILY_API_KEY from .env
                    st.success("✅ Web search enabled")
                except Exception as e:
                    st.warning(f"Web search setup failed: {e}")
            
            # Create agent configuration
            config = AgentConfig()
            config.retrieval_config["vector_threshold"] = vector_thresh
            config.retrieval_config["dataset_threshold"] = dataset_thresh
            
            # Verify RAG components are available
            if not st.session_state.rag_retriever:
                st.error("RAG retriever not available. Please build RAG system first.")
                return
            
            # Create the agent
            agent = LegalQAAgent(
                vector_store=st.session_state.rag_retriever.vectorstore,
                dataset_df=main_df,
                llm=llm,
                web_search_tool=web_search_tool,
                title_index=st.session_state.rag_title_idx,
                config=config.get_config()
            )
            
            st.session_state.agent_system = agent
            st.session_state.agent_ready = True
            st.success("✅ Agent system ready!")
            
    except Exception as e:
        st.error(f"Error setting up agent: {e}")
        import traceback
        st.error(traceback.format_exc())

def _reset_agent():
    """Reset the agent system"""
    st.session_state.agent_system = None
    st.session_state.agent_ready = False
    st.session_state.agent_qa_history = []
    st.info("Agent system reset.")

def agent_qa_interface(selected_provider, selected_model, response_type):
    """Agent Q&A interface integrated into main app"""
    st.subheader("🤖 Multi-Stage Agent Q&A")
    
    if not st.session_state.agent_ready:
        st.info("👈 Please configure and build the Agent system from the sidebar")
        return
    
    # Question input
    question = st.text_area(
        "Ask your legal question:",
        placeholder="e.g., What is the principle of subsidiarity in EU law?",
        height=100,
        key="agent_question_input"
    )
    
    col1, col2, col3 = st.columns([2, 2, 3])
    
    with col1:
        if st.button("🚀 Ask Agent", type="primary", key="ask_agent_btn"):
            if question.strip():
                _process_agent_question(question)
            else:
                st.error("Please enter a question")
    
    with col2:
        if st.button("🗑️ Clear History", key="clear_agent_history"):
            st.session_state.agent_qa_history = []
            st.rerun()
    
    with col3:
        if st.session_state.agent_qa_history:
            if st.button("📥 Export Agent Results", key="export_agent_results"):
                _export_agent_results()
    
    # Display agent status
    if st.session_state.agent_system:
        status = st.session_state.agent_system.get_system_status()
        with st.expander("🔧 Agent System Status"):
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Vector Store", "✅" if status["vector_store_ready"] else "❌")
                st.metric("Dataset Size", f"{status['dataset_size']:,}")
            with col_s2:
                st.metric("Web Search", "✅" if status["web_search_enabled"] else "❌")
                st.metric("Title Index", "✅" if status["title_index_ready"] else "❌")
            with col_s3:
                st.write("**LLM Model:**")
                st.code(status["llm_model"])
    
    # Display Q&A history
    if st.session_state.agent_qa_history:
        st.markdown("---")
        _display_agent_qa_results()

def _process_agent_question(question: str):
    """Process question through the agent system"""
    with st.spinner("🤖 Processing through multi-stage retrieval..."):
        try:
            # Run the agent
            result = asyncio.run(st.session_state.agent_system.answer_question(question))
            
            # Add to history
            st.session_state.agent_qa_history.insert(0, {
                "question": question,
                "result": result,
                "timestamp": datetime.now()
            })
            
            st.success("✅ Question processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing question: {e}")
            import traceback
            st.error(traceback.format_exc())

def _display_agent_qa_results():
    """Display agent Q&A results with detailed breakdown"""
    st.subheader("📋 Agent Q&A Results")
    
    for i, qa in enumerate(st.session_state.agent_qa_history):
        with st.expander(f"Q{i+1}: {qa['question'][:80]}...", expanded=(i==0)):
            result = qa["result"]
            
            # Answer
            st.markdown("**🤖 Agent Answer:**")
            st.info(result["answer"])
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                confidence = result["confidence"]["overall_confidence"]
                st.metric("Overall Confidence", f"{confidence:.3f}")
                
            with col2:
                st.metric("Grounding Level", result["grounding_level"].title())
                
            with col3:
                st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                
            with col4:
                st.metric("Sources Found", result["metadata"]["num_sources"])
            
            # Retrieval stages visualization
            st.markdown("**🔍 Retrieval Stages Used:**")
            stages_used = result["retrieval_stages"]
            stage_colors = {
                "vector_store": "🟢",
                "dataset_query": "🟡", 
                "web_search": "🔵"
            }
            
            if len(stages_used) > 0:
                stage_cols = st.columns(len(stages_used))
                for idx, stage in enumerate(stages_used):
                    with stage_cols[idx]:
                        color = stage_colors.get(stage, "⚪")
                        st.markdown(f"{color} **{stage.replace('_', ' ').title()}**")
            
            # Confidence breakdown
            st.markdown("**📊 Confidence Components:**")
            conf_components = result["confidence"]["components"]
            
            conf_col1, conf_col2, conf_col3 = st.columns(3)
            with conf_col1:
                st.metric("Retrieval", f"{conf_components['retrieval_confidence']:.3f}")
            with conf_col2:
                st.metric("Context Quality", f"{conf_components['context_quality']:.3f}")
            with conf_col3:
                st.metric("Grounding", f"{conf_components['grounding_confidence']:.3f}")
            
            # Sources breakdown
            sources_info = result["sources"]
            if sources_info.get("formatted_sources"):
                st.markdown("**📚 Sources by Type:**")
                
                # Source type counts
                type_counts = sources_info.get("sources_by_type", {})
                if type_counts:
                    type_cols = st.columns(len(type_counts))
                    for idx, (source_type, count) in enumerate(type_counts.items()):
                        with type_cols[idx]:
                            st.metric(source_type.replace('_', ' ').title(), count)
                
                # Detailed sources - use a simple list instead of nested expander
                st.markdown("**📖 Detailed Sources:**")
                for source in sources_info["formatted_sources"]:
                    st.write(f"• {source}")
            
            # Technical details - use simple markdown instead of nested expander
            st.markdown("**🔧 Technical Details:**")
            tech_details = {
                "context_quality": result["context_quality"],
                "sufficiency_score": result.get("sufficiency_score", "N/A"),
                "metadata": result["metadata"]
            }
            
            # Display as formatted text instead of JSON
            st.markdown(f"- **Context Quality Score:** {tech_details['context_quality'].get('score', 'N/A')}")
            st.markdown(f"- **Sufficiency Score:** {tech_details['sufficiency_score']}")
            st.markdown(f"- **Context Length:** {tech_details['metadata']['context_length']}")
            st.markdown(f"- **Question:** {tech_details['metadata']['question'][:100]}...")

def _export_agent_results():
    """Export agent Q&A results"""
    if not st.session_state.agent_qa_history:
        st.warning("No agent results to export")
        return
    
    try:
        # Prepare data for export
        export_data = []
        for qa in st.session_state.agent_qa_history:
            result = qa["result"]
            export_data.append({
                "timestamp": qa["timestamp"].isoformat(),
                "question": qa["question"],
                "answer": result["answer"],
                "grounding_level": result["grounding_level"],
                "overall_confidence": result["confidence"]["overall_confidence"],
                "retrieval_confidence": result["confidence"]["components"]["retrieval_confidence"],
                "context_quality": result["confidence"]["components"]["context_quality"],
                "grounding_confidence": result["confidence"]["components"]["grounding_confidence"],
                "processing_time": result["processing_time"],
                "stages_used": ", ".join(result["retrieval_stages"]),
                "num_sources": result["metadata"]["num_sources"],
                "context_length": result["metadata"]["context_length"]
            })
        
        df = pd.DataFrame(export_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_qa_results_{timestamp}.xlsx"
        
        # Create Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Agent_QA_Results')
        
        st.download_button(
            label="📥 Download Agent Results",
            data=output.getvalue(),
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.success(f"✅ Agent results ready for download: {filename}")
        
    except Exception as e:
        st.error(f"Export failed: {e}")


if __name__ == "__main__":
    main()