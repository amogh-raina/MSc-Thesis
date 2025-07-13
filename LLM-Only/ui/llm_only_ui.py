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
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
import json
from io import BytesIO

# Load environment variables
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

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="LLM Legal Knowledge Evaluator",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ LLM Legal Knowledge Evaluator")
    st.markdown("Evaluate LLM performance on legal questions")
    st.markdown("---")
    
    # Initialize evaluator
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = LLMEvaluator()
        st.session_state.evaluation_history = []
        st.session_state.question_bank_ready = False
        st.session_state.embedding_enabled = False

    # Sidebar configuration
    selected_llm_provider, selected_llm_model, response_type = sidebar_configuration()
    
    # Main content area
    if not st.session_state.question_bank_ready:
        st.info("👆 Please load the question bank from the sidebar to begin evaluation")
    else:
        if selected_llm_provider and selected_llm_model:
            tab1, tab2 = st.tabs([
                "📝 Manual Evaluation",
                "📊 Batch Evaluation"
            ])
            with tab1:
                manual_evaluation_interface(selected_llm_provider, selected_llm_model, response_type)
            with tab2:
                batch_evaluation_interface(selected_llm_provider, selected_llm_model, response_type)
            
        else:
            st.warning("Please select LLM provider and model from sidebar")


def sidebar_configuration():
    """Sidebar configuration"""
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data directory input
        st.subheader("📄 Question Database")
        data_dir = st.text_input(
            "Data Directory Path", 
            value="E:/Projects/MSc-Thesis/JSON Trial 1",
            help="Path to directory containing BEUL_EXAM_*.json files"
        )
        
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
            "Enable embeddings for semantic-similarity",
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
        
        # Model Selection
        st.subheader("🤖 Model Selection")
    

        # Continue with existing provider selection...
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
        
        response_type = st.selectbox(
            "Response Type",
            ["detailed", "concise"],
            help="Choose whether to generate detailed or concise answers"
        )
        
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
        preview_data.append({
        "Question #": i + 1,
        "Year": result.get("year", "N/A"),
        "Q#": result.get("question_number", "N/A"),
        "BLEU": f"{result['evaluation']['bleu_score']:.3f}",
        "ROUGE": f"{result['evaluation']['rouge_score']:.3f}",
        "Str Sim": f"{result['evaluation']['string_similarity_score']:.3f}",
        "Sem Sim": f"{result['evaluation'].get('semantic_similarity_score', float('nan')):.3f}",
        "Question Preview": (result["question"][:50] + "..." if len(result["question"]) > 50 else result["question"])
        })
    
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
def export_results(results: List[Dict], format_type: str = "excel"):
    """Export results to Excel or JSON with proper error handling"""
    if not results:
        st.warning("No results to export")
        return
    
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
                    "string_similarity_score":    result.get("evaluation", {}).get("string_similarity_score", 0),
                    "semantic_similarity_score":  result.get("evaluation", {}).get("semantic_similarity_score", 0),
                    "source_file": result.get("source_file", "")
                }
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

if __name__ == "__main__":
    main()