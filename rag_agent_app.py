import streamlit as st
import pandas as pd
import asyncio
from pathlib import Path
import os
from datetime import datetime

# Import your existing model managers
from rag_app import ModelManager, EmbeddingManager

# Import the new agent system
from Agent.core.agent import LegalQAAgent
from Agent.tools.web_search_tool import WebSearchTool
from Agent.config.agent_config import AgentConfig
from RAG_Pipeline.rag_pipeline import RAGPipeline


def main():
    st.set_page_config(
        page_title="Legal QA Agent",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Legal QA Agent")
    st.markdown("Multi-stage retrieval system for legal question answering")
    
    # Initialize session state
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "agent_ready" not in st.session_state:
        st.session_state.agent_ready = False
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Agent Configuration")
        
        # Model selection
        llm_providers = ModelManager.get_available_llm_providers()
        if llm_providers:
            selected_provider = st.selectbox("LLM Provider", list(llm_providers.keys()))
            selected_model = st.selectbox("LLM Model", llm_providers[selected_provider]["models"])
        else:
            st.error("No LLM providers available")
            return
        
        # Dataset upload
        st.subheader("📊 Dataset Configuration")
        dataset_file = st.file_uploader("Upload Main Dataset (CSV)", type=["csv"])
        vector_dataset_file = st.file_uploader("Upload Vector Dataset (CSV)", type=["csv"])
        chroma_dir = st.text_input("Chroma Directory", "./chroma_db")
        
        # Web search
        st.subheader("🌐 Web Search")
        enable_web_search = st.checkbox("Enable Tavily Web Search")
        if enable_web_search:
            tavily_key = st.text_input("Tavily API Key", type="password")
        
        # Build agent
        if st.button("🔧 Build Agent"):
            if dataset_file and vector_dataset_file:
                build_agent(
                    selected_provider, selected_model,
                    dataset_file, vector_dataset_file, chroma_dir,
                    enable_web_search, tavily_key if enable_web_search else None
                )
            else:
                st.error("Please upload both datasets")
    
    # Main interface
    if st.session_state.agent_ready:
        qa_interface()
    else:
        st.info("👈 Please configure and build the agent from the sidebar")


def build_agent(provider, model, dataset_file, vector_file, chroma_dir, enable_web, tavily_key):
    """Build the agent with all components"""
    with st.spinner("Building agent..."):
        try:
            # Load datasets
            main_df = pd.read_csv(dataset_file)
            vector_df = pd.read_csv(vector_file)
            
            st.write(f"Loaded main dataset: {len(main_df)} rows")
            st.write(f"Loaded vector dataset: {len(vector_df)} rows")
            
            # Create LLM
            llm = ModelManager.create_llm(provider, model)
            if not llm:
                st.error("Failed to create LLM")
                return
            
            # Create embedding model (use a default one)
            embedding_providers = EmbeddingManager.get_available_embedding_providers()
            if "HuggingFace" in embedding_providers:
                embed_model = EmbeddingManager.create_embedding_model(
                    "HuggingFace", 
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            else:
                st.error("No embedding providers available")
                return
            
            # Setup RAG pipeline for vector store
            rag_pipeline = RAGPipeline(
                dataset_path=None,  # We'll pass the dataframe directly
                persist_dir=Path(chroma_dir),
                embedding=embed_model,
                k=7,
                force_rebuild=False
            )
            
            # Initialize with vector dataframe
            rag_pipeline.df = vector_df
            rag_pipeline.initialise()
            
            # Setup web search if enabled
            web_search_tool = None
            if enable_web and tavily_key:
                try:
                    web_search_tool = WebSearchTool(tavily_key)
                    st.success("✅ Web search enabled")
                except Exception as e:
                    st.warning(f"Web search setup failed: {e}")
            
            # Create agent
            agent = LegalQAAgent(
                vector_store=rag_pipeline.retriever.vectorstore,
                dataset_df=main_df,
                llm=llm,
                web_search_tool=web_search_tool,
                title_index=rag_pipeline.title_index,
                config=AgentConfig().get_config()
            )
            
            st.session_state.agent = agent
            st.session_state.agent_ready = True
            st.success("✅ Agent built successfully!")
            
        except Exception as e:
            st.error(f"Error building agent: {e}")


def qa_interface():
    """Main Q&A interface"""
    st.subheader("💬 Ask a Legal Question")
    
    # Question input
    question = st.text_area(
        "Enter your legal question:",
        placeholder="e.g., What is the principle of subsidiarity in EU law?",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🚀 Ask Question", type="primary"):
            if question.strip():
                answer_question(question)
            else:
                st.error("Please enter a question")
    
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.qa_history = []
            st.rerun()
    
    # Display results
    if st.session_state.qa_history:
        st.markdown("---")
        display_qa_results()


def answer_question(question: str):
    """Process question through the agent"""
    with st.spinner("Processing question through multi-stage retrieval..."):
        try:
            # Run the agent
            result = asyncio.run(st.session_state.agent.answer_question(question))
            
            # Add to history
            st.session_state.qa_history.insert(0, {
                "question": question,
                "result": result,
                "timestamp": datetime.now()
            })
            
            st.success("✅ Question processed!")
            
        except Exception as e:
            st.error(f"Error processing question: {e}")


def display_qa_results():
    """Display Q&A results with detailed breakdown"""
    st.subheader("📋 Q&A Results")
    
    for i, qa in enumerate(st.session_state.qa_history):
        with st.expander(f"Q{i+1}: {qa['question'][:100]}...", expanded=(i==0)):
            result = qa["result"]
            
            # Answer
            st.markdown("**🤖 Answer:**")
            st.info(result["answer"])
            
            # Metadata in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Confidence", f"{result['confidence']['overall_confidence']:.2f}")
            
            with col2:
                st.metric("Grounding Level", result["grounding_level"])
            
            with col3:
                st.metric("Processing Time", f"{result['processing_time']:.2f}s")
            
            with col4:
                st.metric("Sources", result["metadata"]["num_sources"])
            
            # Retrieval stages
            st.markdown("**🔍 Retrieval Stages Used:**")
            for stage in result["retrieval_stages"]:
                st.badge(stage, type="secondary")
            
            # Confidence breakdown
            st.markdown("**📊 Confidence Breakdown:**")
            conf_components = result["confidence"]["components"]
            conf_df = pd.DataFrame([
                {"Component": "Retrieval", "Score": conf_components["retrieval_confidence"]},
                {"Component": "Context Quality", "Score": conf_components["context_quality"]},
                {"Component": "Grounding", "Score": conf_components["grounding_confidence"]}
            ])
            st.bar_chart(conf_df.set_index("Component"))
            
            # Sources
            if result["sources"]["formatted_sources"]:
                st.markdown("**📚 Sources:**")
                for source in result["sources"]["formatted_sources"]:
                    st.write(f"• {source}")
            
            # Raw metadata (collapsible)
            with st.expander("🔧 Technical Details"):
                st.json(result["metadata"])


if __name__ == "__main__":
    main() 