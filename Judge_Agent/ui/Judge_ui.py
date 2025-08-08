# Streamlit UI for Judge Agent 
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
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Import shared components from Main folder
from Main.core.model_manager import ModelManager
from Main.config.settings import *

# Import Judge Agent components
sys.path.append(str(project_root / "Judge_Agent"))
from workflow import run_judge_evaluation
from llm_judge_evaluator import LLMJudgeEvaluator
from Judge_Agent.utils import load_file

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Judge Agent - Legal Answer Evaluator",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Judge Agent - Legal Answer Evaluator")
    st.markdown("Autonomous evaluation of legal answers with post-model guardrails and web search capability")
    st.markdown("---")
    
    # Initialize session state
    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = []

    # Sidebar configuration
    selected_llm_provider, selected_llm_model = sidebar_configuration()
    
    # Main content area
    if selected_llm_provider and selected_llm_model:
        tab1, tab2, tab3 = st.tabs([
            "📁 File Upload & Evaluation",
            "📊 Results & Export",
            "🔧 Quality Analysis"
        ])
        with tab1:
            file_evaluation_interface(selected_llm_provider, selected_llm_model)
        with tab2:
            results_interface()
        with tab3:
            quality_analysis_interface()
    else:
        st.warning("Please select LLM provider and model from sidebar")

def sidebar_configuration():
    """Sidebar configuration"""
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        
        llm_providers = ModelManager.get_available_llm_providers()
        
        if not llm_providers:
            st.error("❌ No LLM providers available. Please check your API keys.")
            return None, None
        
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
                help="Select the specific model to use for evaluation"
            )
        
        st.markdown("---")
        
        # Evaluation Method Selection
        st.subheader("⚖️ Evaluation Method")
        
        evaluation_method = st.radio(
            "Choose evaluation approach:",
            options=["Agent Judge", "LLM Judge"],
            help="""
            • **Agent Judge**: Autonomous agent with web search capabilities and post-model guardrails
            • **LLM Judge**: Pure parametric knowledge evaluation (no web search, faster)
            """,
            index=0  # Default to Agent Judge
        )
        
        # Store in session state for use in evaluation
        st.session_state.evaluation_method = evaluation_method
        
        if evaluation_method == "Agent Judge":
            st.info("🤖 **Agent Judge**: Uses autonomous agent with web search and guardrails")
        else:
            st.info("🧠 **LLM Judge**: Uses parametric knowledge only (no web search)")
        
        st.markdown("---")
        
        # API Status
        st.subheader("🔧 API Status")
        
        # Check API keys
        api_status = []
        if selected_llm_provider:
            if selected_llm_provider == "OpenAI":
                api_status.append(("OpenAI", "✅" if os.getenv("OPENAI_API_KEY") else "❌"))
            elif selected_llm_provider == "NVIDIA":
                api_status.append(("NVIDIA", "✅" if os.getenv("NVIDIA_API_KEY") else "❌"))
            elif selected_llm_provider == "Mistral":
                api_status.append(("Mistral", "✅" if os.getenv("MISTRAL_API_KEY") else "❌"))
            elif selected_llm_provider == "Google":
                api_status.append(("Google", "✅" if os.getenv("GOOGLE_API_KEY") else "❌"))
            elif selected_llm_provider == "Github":
                api_status.append(("GitHub", "✅" if os.getenv("GITHUB_TOKEN") else "❌"))
        
        api_status.append(("Tavily (Web Search)", "✅" if os.getenv("TAVILY_API_KEY") else "❌"))
        
        for service, status in api_status:
            st.write(f"{service}: {status}")
        
        st.markdown("---")
        
        # Current Session Stats
        st.subheader("📊 Session Stats")
        if st.session_state.evaluation_results:
            all_results = st.session_state.evaluation_results
            total_evals = len(all_results)
            
            # Determine current model from most recent evaluation
            current_model_provider = None
            current_model_name = None
            if all_results:
                latest_result = all_results[-1]
                current_model_provider = latest_result.get("llm_provider")
                current_model_name = latest_result.get("llm_model")
            
            # Filter for current model
            current_model_results = []
            if current_model_provider and current_model_name:
                current_model_results = [
                    r for r in all_results 
                    if r.get("llm_provider") == current_model_provider 
                    and r.get("llm_model") == current_model_name
                ]
            
            # Total session stats
            st.metric("Total Session Evaluations", total_evals)
            
            # Current model stats
            if current_model_results:
                st.write(f"**Current Model: {current_model_provider}/{current_model_name}**")
                current_successful = [r for r in current_model_results if "scores" in r and "error" not in r.get("scores", {})]
                current_guardrails = sum(1 for r in current_successful if r.get("fixes_applied"))
                current_web_searches = sum(1 for r in current_successful if r.get("web_search_used"))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Model Evals", len(current_model_results))
                    st.metric("Success Rate", f"{len(current_successful)/len(current_model_results)*100:.1f}%" if current_model_results else "0%")
                with col2:
                    st.metric("Web Searches", current_web_searches)
                    st.metric("Guardrails Applied", current_guardrails)
            
            # Show configuration breakdown if multiple configurations used
            config_counts = {}
            for result in all_results:
                method_icon = "🤖" if result.get("evaluation_method") == "Agent Judge" else "🧠"
                config_key = f"{result.get('llm_provider', 'Unknown')}/{result.get('llm_model', 'Unknown')} - {method_icon}{result.get('evaluation_method', 'Unknown')}"
                config_counts[config_key] = config_counts.get(config_key, 0) + 1
            
            if len(config_counts) > 1:
                st.write("**All Configurations Used:**")
                current_config_icon = "🤖" if st.session_state.get('evaluation_method', 'Agent Judge') == "Agent Judge" else "🧠"
                current_config_key = f"{current_model_provider}/{current_model_name} - {current_config_icon}{st.session_state.get('evaluation_method', 'Agent Judge')}"
                for config, count in config_counts.items():
                    indicator = "👈" if config == current_config_key else ""
                    st.write(f"• {config}: {count} {indicator}")
            
            st.info("💡 Results tab shows current configuration only, export downloads all configurations")
        else:
            st.text("No evaluations yet")
        
        return selected_llm_provider, selected_llm_model

def file_evaluation_interface(selected_llm_provider, selected_llm_model):
    """File upload and evaluation interface"""
    
    st.subheader("📁 Upload & Evaluate")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Upload CSV/Excel with: `question`, `reference_answer`, `generated_answer` columns")
        
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=['csv', 'xlsx']
        )
        
        if uploaded_file is not None:
            # Preview the file
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.subheader("📋 File Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Validate columns
                required_columns = ['question', 'reference_answer', 'generated_answer']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {missing_columns}")
                    st.write(f"Available columns: {list(df.columns)}")
                else:
                    st.success(f"✅ File validated - {len(df)} rows ready for evaluation")
                    
                    # Evaluation button
                    if st.button("🚀 Start Evaluation", type="primary"):
                        run_evaluation(uploaded_file, df, selected_llm_provider, selected_llm_model)
                        
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        st.subheader("ℹ️ Evaluation Workflow")
        
        # Show different workflow based on selected method
        evaluation_method = st.session_state.get('evaluation_method', 'Agent Judge')
        
        if evaluation_method == "Agent Judge":
            st.info("🤖 **Agent Judge**: Autonomous evaluation with web search, guardrails, and context optimization")
        else:
            st.info("🧠 **LLM Judge**: Fast parametric evaluation with individual metric prompts (no web search)")
        
        st.write("**Rubric**: Accuracy, Completeness, Relevance, Overall (1-5 scale)")

def run_llm_judge_evaluation(file_path: str, llm, provider: str, model: str) -> Dict[str, Any]:
    """
    Run LLM Judge evaluation using parametric knowledge only
    
    Args:
        file_path: Path to the file with trios
        llm: LLM instance
        provider: LLM provider name
        model: LLM model name
        
    Returns:
        Dictionary with evaluation results in same format as agent judge
    """
    try:
        # Load file using the same utility as agent judge
        all_trios = load_file(file_path)
        if not all_trios:
            return {
                "error": "No valid trios found in file",
                "all_evaluations": []
            }
        
        # Create LLM judge evaluator
        llm_judge = LLMJudgeEvaluator(llm)
        
        evaluations = []
        
        for i, trio in enumerate(all_trios):
            try:
                # Run evaluation for this trio
                evaluation_result = llm_judge.evaluate_all(
                    question=trio["question"],
                    reference_answer=trio["reference_answer"],
                    generated_answer=trio["generated_answer"]
                )
                
                # Format result to match agent judge structure
                result = {
                    "trio_index": i,
                    "question": trio["question"],
                    "reference_answer": trio["reference_answer"],  # Add reference answer
                    "generated_answer": trio["generated_answer"],  # Add generated answer
                    "scores": {
                        "accuracy": evaluation_result["correctness"]["score"],
                        "accuracy_reason": evaluation_result["correctness"]["explanation"],
                        "completeness": evaluation_result["groundedness"]["score"],
                        "completeness_reason": evaluation_result["groundedness"]["explanation"],
                        "relevance": evaluation_result["relevance"]["score"],
                        "relevance_reason": evaluation_result["relevance"]["explanation"],
                        "overall": evaluation_result["retrieval_relevance"]["score"],
                        "overall_reason": evaluation_result["retrieval_relevance"]["explanation"]
                    },
                    "web_search_used": False,  # LLM judge never uses web search
                    "search_results": None,
                    "evaluation_method": "LLM Judge",
                    "llm_provider": provider,
                    "llm_model": model,
                    "timestamp": datetime.now().isoformat()
                }
                
                evaluations.append(result)
                
            except Exception as e:
                # Handle individual evaluation errors
                error_result = {
                    "trio_index": i,
                    "question": trio["question"],
                    "reference_answer": trio["reference_answer"],  # Add reference answer
                    "generated_answer": trio["generated_answer"],  # Add generated answer
                    "error": f"LLM Judge evaluation failed: {str(e)}",
                    "scores": {"error": "Evaluation failed"},
                    "evaluation_method": "LLM Judge",
                    "llm_provider": provider,
                    "llm_model": model,
                    "timestamp": datetime.now().isoformat()
                }
                evaluations.append(error_result)
        
        return {
            "all_evaluations": evaluations,
            "is_complete": True
        }
        
    except Exception as e:
        return {
            "error": f"LLM Judge evaluation failed: {str(e)}",
            "all_evaluations": []
        }

def run_evaluation(uploaded_file, df, provider, model):
    """Run the evaluation workflow with enhanced error handling"""
    
    # Save uploaded file temporarily
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Get evaluation method from session state
    evaluation_method = st.session_state.get('evaluation_method', 'Agent Judge')
    
    # Create LLM instance (ModelManager now has comprehensive error handling for both LLMs and embeddings)
    with st.spinner("🤖 Initializing model..."):
        llm = ModelManager.create_llm(provider, model)
        if not llm:
            st.error(f"❌ Failed to create LLM instance for {provider}/{model}")
            st.info("💡 Check console for detailed debugging information. Try running as administrator or using a different provider.")
            return
    
    try:
        # Run evaluation with enhanced feedback based on chosen method
        if evaluation_method == "Agent Judge":
            with st.spinner(f"🤖 Running Agent Judge evaluation on {len(df)} question-answer pairs..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Running Agent Judge workflow with web search capabilities...")
                
                # Run the agent workflow
                final_state = run_judge_evaluation(temp_file_path, llm)
                progress_bar.progress(1.0)
                
                # Extract and process results
                evaluations = final_state.get("all_evaluations", [])
                error_info = final_state.get("error")
                
                if error_info:
                    st.error(f"❌ Agent Judge workflow error: {error_info}")
                    return
                
                # Add metadata to results
                for eval_result in evaluations:
                    eval_result.update({
                        "evaluation_method": "Agent Judge",
                        "llm_provider": provider,
                        "llm_model": model,
                        "timestamp": datetime.now().isoformat()
                    })
                
                progress_bar.empty()
                status_text.empty()
        
        else:  # LLM Judge
            with st.spinner(f"🧠 Running LLM Judge evaluation on {len(df)} question-answer pairs..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Running LLM Judge evaluation with parametric knowledge...")
                
                # Run the LLM judge workflow
                final_state = run_llm_judge_evaluation(temp_file_path, llm, provider, model)
                progress_bar.progress(1.0)
                
                # Extract and process results
                evaluations = final_state.get("all_evaluations", [])
                error_info = final_state.get("error")
                
                if error_info:
                    st.error(f"❌ LLM Judge workflow error: {error_info}")
                    return
                
                progress_bar.empty()
                status_text.empty()
        
        if evaluations:
            # Store in session state
            st.session_state.evaluation_results.extend(evaluations)
            
            st.success(f"✅ {evaluation_method} evaluation completed! Processed {len(evaluations)} evaluations")
            
            # Show enhanced summary BEFORE any UI refresh
            display_enhanced_summary(evaluations, evaluation_method)
            
            # DON'T call st.rerun() here - it clears the summary!
            # The results interface will update automatically when session state changes
            
        else:
            st.error("❌ No evaluations completed")
            
    except Exception as e:
        st.error(f"❌ Evaluation failed: {str(e)}")
        # Show more detailed error info
        with st.expander("🔍 Error Details"):
            st.exception(e)
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def display_enhanced_summary(evaluations, evaluation_method="Agent Judge"):
    """Display enhanced summary with method-specific information"""
    
    st.subheader(f"📊 {evaluation_method} Evaluation Summary")
    
    successful = [e for e in evaluations if "scores" in e and "error" not in e.get("scores", {})]
    errors = len(evaluations) - len(successful)
    
    if evaluation_method == "Agent Judge":
        # Agent Judge specific metrics
        web_searches = sum(1 for e in successful if e.get("web_search_used"))
        guardrails_applied = sum(1 for e in successful if e.get("fixes_applied"))
        quality_warnings = sum(1 for e in successful if e.get("validation_warnings"))
        
        # Main metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Processed", len(evaluations))
        with col2:
            st.metric("Successful", len(successful))
        with col3:
            st.metric("Web Searches", web_searches)
        with col4:
            st.metric("Guardrails Applied", guardrails_applied)
        with col5:
            st.metric("Quality Warnings", quality_warnings)
        
        if successful:
            search_rate = web_searches / len(successful) * 100 if successful else 0
            st.success(f"🤖 {len(successful)} Agent Judge evaluations completed. Web search used: {search_rate:.0f}%")
    
    else:  # LLM Judge
        # LLM Judge specific metrics
        parametric_evals = len(successful)  # All successful are parametric
        
        # Main metrics  
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Processed", len(evaluations))
        with col2:
            st.metric("Successful", len(successful))
        with col3:
            st.metric("Parametric Evals", parametric_evals)
        with col4:
            st.metric("Error Rate", f"{(errors/len(evaluations)*100):.1f}%" if evaluations else "0%")
        
        if successful:
            # Calculate average overall score
            overall_scores = [result.get("scores", {}).get("overall", 0) for result in successful if result.get("scores", {}).get("overall")]
            avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
            st.success(f"🧠 {len(successful)} LLM Judge evaluations completed. Average overall score: {avg_overall:.1f}/5.0")

def quality_analysis_interface():
    """Quality analysis and guardrails monitoring interface"""
    
    st.subheader("🔧 Quality Analysis")
    
    if not st.session_state.evaluation_results:
        st.info("No evaluation results yet.")
        return
    
    results = st.session_state.evaluation_results
    successful = [r for r in results if "scores" in r and "error" not in r.get("scores", {})]
    
    if not successful:
        st.warning("No successful evaluations to analyze.")
        return
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    
    guardrails_results = [r for r in successful if r.get("fixes_applied")]
    warning_results = [r for r in successful if r.get("validation_warnings")]
    agent_results = [r for r in successful if r.get("evaluation_method") == "Agent Judge"]
    
    with col1:
        st.metric("Guardrails Applied", f"{len(guardrails_results)}/{len(successful)}")
    with col2:
        st.metric("Quality Warnings", f"{len(warning_results)}/{len(successful)}")
    with col3:
        st.metric("Agent Judge", f"{len(agent_results)}/{len(successful)}")
    
    # Score averages
    if successful:
        score_data = {}
        for result in successful:
            for metric, value in result["scores"].items():
                if isinstance(value, (int, float)):
                    if metric not in score_data:
                        score_data[metric] = []
                    score_data[metric].append(value)
        
        if score_data:
            st.write("**Average Scores:**")
            cols = st.columns(len(score_data))
            for i, (metric, values) in enumerate(score_data.items()):
                with cols[i]:
                    avg = sum(values) / len(values)
                    st.metric(metric.replace('_', ' ').title(), f"{avg:.2f}")
    

def results_interface():
    """Enhanced results viewing and export interface - shows current model results, exports all"""
    
    st.subheader("📊 Evaluation Results")
    
    if not st.session_state.evaluation_results:
        st.info("No evaluation results yet. Upload a file and run evaluation first.")
        return
    
    # Get current model selection from sidebar (we need to get this from the sidebar state)
    # Since we can't directly access sidebar state here, we'll use the last evaluation's model info
    # as the "current model" - this works because evaluations are run immediately after model selection
    all_results = st.session_state.evaluation_results
    
    # Determine current model and method from the most recent evaluation
    current_model_provider = None
    current_model_name = None
    current_evaluation_method = None
    if all_results:
        latest_result = all_results[-1]
        current_model_provider = latest_result.get("llm_provider")
        current_model_name = latest_result.get("llm_model")
        current_evaluation_method = latest_result.get("evaluation_method", "Agent Judge")
    
    # Filter results for current model and method (for display)
    current_config_results = []
    if current_model_provider and current_model_name and current_evaluation_method:
        current_config_results = [
            r for r in all_results 
            if r.get("llm_provider") == current_model_provider 
            and r.get("llm_model") == current_model_name
            and r.get("evaluation_method") == current_evaluation_method
        ]
    
    # Results summary
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Current Configuration Results")
        
        if current_model_provider and current_model_name and current_evaluation_method:
            method_icon = "🤖" if current_evaluation_method == "Agent Judge" else "🧠"
            st.info(f"{method_icon} Showing results for: **{current_model_provider}/{current_model_name}** - **{current_evaluation_method}** ({len(current_config_results)} evaluations)")
            
            if len(all_results) > len(current_config_results):
                st.caption(f"💡 Total session results: {len(all_results)} evaluations (across all configurations)")
        
        # Create results dataframe from CURRENT CONFIGURATION results only
        if current_config_results:
            table_data = []
            for i, result in enumerate(current_config_results):
                row = {
                    "Index": i + 1,
                    "Question": result.get("question", "")[:100] + "..." if len(result.get("question", "")) > 100 else result.get("question", ""),
                    "Method": "🤖" if result.get("evaluation_method") == "Agent Judge" else "🧠",
                    "Web Search": "✅" if result.get("web_search_used") else "❌",
                    "Guardrails": "🔧" if result.get("fixes_applied") else "✅",
                    "Warnings": "⚠️" if result.get("validation_warnings") else "✅",
                }
                
                # Add scores and reasoning if available
                if "scores" in result:
                    for metric, value in result["scores"].items():
                        if isinstance(value, (int, float)):
                            row[metric.replace('_', ' ').title()] = f"{value}"
                        elif isinstance(value, str) and metric.endswith('_reason'):
                            # Add reasoning columns (truncated for table display)
                            reason_display = value[:60] + "..." if len(value) > 60 else value
                            row[metric.replace('_', ' ').title()] = reason_display
                
                table_data.append(row)
            
            df_results = pd.DataFrame(table_data)
            # Use dynamic key to force refresh when results change
            table_key = f"current_results_table_{len(current_config_results)}_{hash(str(current_config_results[-1].get('timestamp', '')))}"
            st.dataframe(df_results, use_container_width=True, key=table_key)
            
            # Add detailed results expander
            with st.expander("🔍 Detailed Results Analysis", expanded=False):
                st.markdown("**Complete evaluation details with full reasoning**")
                
                # Allow user to select which result to view in detail
                if len(current_config_results) > 1:
                    selected_idx = st.selectbox(
                        "Select result to view in detail:",
                        range(len(current_config_results)),
                        format_func=lambda x: f"Result {x+1}: {current_config_results[x].get('question', '')[:50]}..."
                    )
                else:
                    selected_idx = 0
                
                if selected_idx < len(current_config_results):
                    detailed_result = current_config_results[selected_idx]
                    
                    # Show evaluation metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Evaluation Method", detailed_result.get("evaluation_method", "Unknown"))
                        st.metric("Web Search Used", "✅ Yes" if detailed_result.get("web_search_used") else "❌ No")
                    with col2:
                        st.metric("LLM Provider", detailed_result.get("llm_provider", "Unknown"))
                        st.metric("Guardrails Applied", "🔧 Yes" if detailed_result.get("fixes_applied") else "✅ No")
                    with col3:
                        st.metric("LLM Model", detailed_result.get("llm_model", "Unknown"))
                        st.metric("Quality Warnings", "⚠️ Yes" if detailed_result.get("validation_warnings") else "✅ No")
                    
                    # Show question and answers
                    st.subheader("📝 Question & Answers")
                    st.write("**Question:**")
                    st.write(detailed_result.get("question", "No question available"))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Reference Answer:**")
                        st.text_area("", detailed_result.get("reference_answer", "No reference answer available"), height=150, key=f"ref_{selected_idx}")
                    with col2:
                        st.write("**Generated Answer:**")
                        st.text_area("", detailed_result.get("generated_answer", "No generated answer available"), height=150, key=f"gen_{selected_idx}")
                    
                    # Show detailed scores and reasoning
                    st.subheader("📊 Evaluation Scores & Reasoning")
                    if "scores" in detailed_result:
                        scores = detailed_result["scores"]
                        
                        # Group scores with their reasoning
                        score_pairs = [
                            ("Accuracy", scores.get("accuracy"), scores.get("accuracy_reason")),
                            ("Completeness", scores.get("completeness"), scores.get("completeness_reason")),
                            ("Relevance", scores.get("relevance"), scores.get("relevance_reason")),
                            ("Overall", scores.get("overall"), scores.get("overall_reason"))
                        ]
                        
                        for metric_name, score, reasoning in score_pairs:
                            if score is not None:
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.metric(f"{metric_name} Score", f"{score}/5")
                                with col2:
                                    st.write(f"**{metric_name} Reasoning:**")
                                    st.write(reasoning or "No reasoning provided")
                                st.divider()
                    
                    # Show search results if available
                    if detailed_result.get("web_search_used") and detailed_result.get("search_results"):
                        st.subheader("🔍 Web Search Results")
                        with st.expander("View search results", expanded=False):
                            st.text(detailed_result.get("search_results", "No search results available"))
                    
                    # Show guardrails info if available
                    if detailed_result.get("fixes_applied") or detailed_result.get("validation_warnings"):
                        st.subheader("🛡️ Guardrails & Quality Control")
                        
                        if detailed_result.get("fixes_applied"):
                            st.write("**Fixes Applied:**")
                            for fix in detailed_result.get("fixes_applied", []):
                                st.write(f"• {fix}")
                        
                        if detailed_result.get("validation_warnings"):
                            st.write("**Quality Warnings:**")
                            for warning in detailed_result.get("validation_warnings", []):
                                st.write(f"⚠️ {warning}")
        else:
            st.info("No results for current configuration.")
    
    with col2:
        st.subheader("📥 Export Options")
        
        # Show export stats
        if all_results:
            st.metric("Total Session Results", len(all_results))
            
            # Show breakdown by configuration
            config_counts = {}
            for result in all_results:
                method_icon = "🤖" if result.get("evaluation_method") == "Agent Judge" else "🧠"
                config_key = f"{result.get('llm_provider', 'Unknown')}/{result.get('llm_model', 'Unknown')} - {method_icon}{result.get('evaluation_method', 'Unknown')}"
                config_counts[config_key] = config_counts.get(config_key, 0) + 1
            
            if len(config_counts) > 1:
                st.write("**Results by Configuration:**")
                for config, count in config_counts.items():
                    st.write(f"• {config}: {count}")
        
        export_format = st.selectbox("Export Format", ["JSON", "Excel"])
        
        if st.button("📥 Export All Session Results"):
            export_enhanced_results(all_results, export_format.lower())  # Export ALL results
        
        if st.button("🗑️ Clear All Results"):
            st.session_state.evaluation_results = []
            st.success("All results cleared!")
            st.rerun()
    


def export_enhanced_results(results: List[Dict], format_type: str = "json"):
    """Export enhanced results with guardrails information"""
    if not results:
        st.warning("No results to export")
        return
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "excel":
            # Prepare enhanced data for Excel export
            df_data = []
            for result in results:
                # Initialize row with core fields only
                row = {
                    "trio_index": result.get("trio_index", ""),
                    "question": result.get("question", ""),
                    "reference_answer": result.get("reference_answer", ""),  # Add missing reference answer
                    "generated_answer": result.get("generated_answer", ""),  # Add missing generated answer
                    "evaluation_method": result.get("evaluation_method", "Unknown"),
                    "llm_provider": result.get("llm_provider", ""),
                    "llm_model": result.get("llm_model", ""),
                    "web_search_used": result.get("web_search_used", False),
                    "guardrails_applied": bool(result.get("fixes_applied")),
                    "quality_warnings": bool(result.get("validation_warnings")),
                    "fixes_applied": "; ".join(result.get("fixes_applied", [])),
                    "validation_warnings": "; ".join(result.get("validation_warnings", [])),
                    "timestamp": result.get("timestamp", "")
                }
                
                # Add only the expected score fields to avoid contamination
                if "scores" in result and isinstance(result["scores"], dict):
                    expected_score_fields = {
                        "accuracy", "accuracy_reason", 
                        "completeness", "completeness_reason",
                        "relevance", "relevance_reason",
                        "overall", "overall_reason"
                    }
                    
                    for metric, value in result["scores"].items():
                        # Only include expected score fields to avoid search result contamination
                        if metric in expected_score_fields:
                            row[f"score_{metric}"] = value
                        else:
                            # Log unexpected fields for debugging
                            if metric not in ["error"]:  # Don't log error field
                                print(f"⚠️  Skipping unexpected score field: {metric}")
                
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            filename = f"judge_agent_enhanced_results_{timestamp}.xlsx"
            
            # Create Excel file in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Judge_Results')
            
            # Provide download button
            st.download_button(
                label="📥 Download Enhanced Excel Report",
                data=output.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        else:  # JSON format
            filename = f"judge_agent_enhanced_results_{timestamp}.json"
            
            # Create enhanced JSON structure with method breakdown
            agent_judge_results = [r for r in results if r.get("evaluation_method") == "Agent Judge"]
            llm_judge_results = [r for r in results if r.get("evaluation_method") == "LLM Judge"]
            
            enhanced_export = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_evaluations": len(results),
                    "successful_evaluations": len([r for r in results if "scores" in r]),
                    "agent_judge_evaluations": len(agent_judge_results),
                    "llm_judge_evaluations": len(llm_judge_results),
                    "guardrails_applied": len([r for r in results if r.get("fixes_applied")]),
                    "quality_warnings": len([r for r in results if r.get("validation_warnings")]),
                    "web_searches_used": len([r for r in results if r.get("web_search_used")])
                },
                "evaluations": results
            }
            
            json_data = json.dumps(enhanced_export, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Download Enhanced JSON Report",
                data=json_data,
                file_name=filename,
                mime="application/json"
            )
        
        st.success(f"✅ Enhanced export ready for download!")
    
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

if __name__ == "__main__":
    main() 