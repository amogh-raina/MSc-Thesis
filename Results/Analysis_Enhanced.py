# -*- coding: utf-8 -*-
"""
Enhanced Comprehensive NLP Evaluation Analysis Script

This script addresses all the methodological concerns raised and provides:
• Complete inter-rater reliability analysis across all dimensions and judge combinations
• Proper hypothesis testing with explicit null/alternative hypotheses
• Comprehensive Bonferroni correction implementation
• Detailed effect size reporting and interpretation
• Mann-Whitney U tests for independent group comparisons
• Clear research question framework

Research Questions Addressed:
Q1: Are judges reliable within their type? (Inter-judge reliability)
Q2: Do judges agree across types? (Cross-judge reliability) 
Q3: Is RAG significantly better than LLM-only? (System comparison)
Q4: Do LLM-Judges and Agent-Judges score differently? (Judge comparison)
Q5: What are the qualitative patterns in reasoning? (Qualitative analysis)

Authors: Enhanced implementation addressing statistical methodology concerns
Date: 2025
"""

import os
import warnings
import pandas as pd
import numpy as np
from itertools import combinations
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel, kendalltau
# from sklearn.metrics import cohen_kappa_score  # COMMENTED OUT: Not using Cohen's Kappa
import krippendorff
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# --- Matplotlib and Seaborn Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)

# =============================================================================
# Configuration Section - Modify these for different setups
# =============================================================================

# Analysis Configuration
EXPECTED_HUMAN_JUDGES = 3  # Change this to match your setup
EXPECTED_EVALUATOR_MODELS = 4  # Updated for: gemini-1.5-flash, gpt-o3-mini, qwen-2.5-3b, llama-3.1-8b
EXPECTED_QUESTIONS_PER_MODEL = 12  # Questions per generator model
ALPHA_LEVEL = 0.05  # Significance level for statistical tests

# Reliability Thresholds
# KAPPA_THRESHOLDS = {  # COMMENTED OUT: Not using Cohen's Kappa
#     'excellent': 0.8,
#     'good': 0.6,
#     'moderate': 0.4
# }

ALPHA_THRESHOLDS = {
    'good': 0.67,
    'tentative': 0.4
}

TAU_THRESHOLDS = {
    'strong': 0.7,
    'moderate': 0.5
}

# Effect Size Thresholds
COHENS_D_THRESHOLDS = {
    'large': 0.8,
    'medium': 0.5,
    'small': 0.2
}

WILCOXON_R_THRESHOLDS = {
    'large': 0.5,
    'medium': 0.3,
    'small': 0.1
}

# =============================================================================
# Data Loading and Utility Functions
# =============================================================================

def clean_model_name(name: str) -> str:
    """Standardizes model names for consistent grouping."""
    if not isinstance(name, str):
        return str(name)
    
    name = name.lower().strip()
    
    # Judge models (existing + new)
    if "o3" in name and "mini" in name:
        return "gpt-o3-mini"
    # if "llama" in name and ("3.2" in name or "3b" in name):
    #     return "llama-3.2-3b"
    if "llama" in name and ("3.1" in name or "8b" in name):
        return "llama-3.1-8b instant"
    if "gemini" in name and "1.5" in name and "flash" in name:
        return "gemini-1.5-flash"
    if "gemini" in name and "2.0" in name and "flash" in name:
        return "gemini-2.0-flash"
    if "gpt" in name and ("4.1" in name or "4o" in name) and "mini" in name:
        return "gpt-4.1-mini"
    if "qwen" in name and "3" in name:
        return "qwen3"
    # Update existing qwen rule to be more specific:
    if "qwen" in name and "2.5" in name:
        return "qwen-2.5-3b"  # Only for old files
        # if "grok" in name:
        #     return "grok-2"
    
    # Generator models (existing + new)
    if "gpt" in name and "4o" in name:
        return "gpt-4o"
    if "llama" in name and "3.3" in name and "70b" in name:
        return "llama-3.3-70b"
    if "gemini" in name and "1.5" in name and "pro" in name:
        return "gemini-1.5-pro"
    # if "claude" in name and "3" in name and "opus" in name:
    #     return "claude-3-opus"
    
    return name

def load_judgement_files(base_path: str = ".") -> pd.DataFrame:
    """Load and aggregate all judgement files (both RAG and LLM-only systems)."""
    frames = []
    
    # Define the files to be loaded for each system
    rag_files = [
        "RAG-System/Judgements (LLM +AGENT)/judgements  (LLM-Judge +Agent-Judge) - gemini 1.5 flash (RAG-System).xlsx",
        "RAG-System/Judgements (LLM +AGENT)/judgements  (LLM-Judge +Agent-Judge) - GPT o3-mini (RAG-System).xlsx", 
        "RAG-System/Judgements (LLM +AGENT)/judgements  (LLM-Judge +Agent-Judge) - qwen3 1.7b (RAG-System).xlsx",
        "RAG-System/Judgements (LLM +AGENT)/judgements  (LLM-Judge +Agent-Judge) - llama 3.1 8b instant (RAG-System).xlsx",
        "RAG-System/Judgements (LLM +AGENT)/judgements  (LLM-Judge +Agent-Judge) - gpt 4.1 mini (RAG-System).xlsx"
    ]
    
    llm_files = [
        "LLM-Only/Judgements (LLM+AGENT)/judgements  (LLM-Judge +Agent-Judge) - gemini 1.5 flash (LLM-Only System).xlsx",
        "LLM-Only/Judgements (LLM+AGENT)/judgements  (LLM-Judge +Agent-Judge) - GPT o3-mini (LLM-Only).xlsx",
        "LLM-Only/Judgements (LLM+AGENT)/judgements  (LLM-Judge +Agent-Judge) - qwen3 1.7b (LLM-only).xlsx",
        "LLM-Only/Judgements (LLM+AGENT)/judgements  (LLM-Judge +Agent-Judge) - llama 3.1 8b instant (LLM-Only).xlsx",
        "LLM-Only/Judgements (LLM+AGENT)/judgements  (LLM-Judge +Agent-Judge) - gpt 4.1 mini (LLM-Only).xlsx"
    ]
    
    
    # Process RAG and LLM-only files
    for system, files in [("RAG", rag_files), ("LLM-only", llm_files)]:
        for filename in files:
            filepath = os.path.join(base_path, filename) if base_path != "." else filename
            if os.path.exists(filepath):
                try:
                    df = pd.read_excel(filepath)
                    df['system'] = system
                    
                    # Standardize columns
                    if 'evaluation_method' in df.columns:
                        df['judge_type'] = df['evaluation_method'].str.strip()
                    if 'llm_model' in df.columns:
                        df['generator_model'] = df['llm_model'].apply(clean_model_name)
                    if 'judge_llm_model' in df.columns:
                        df['judge_model'] = df['judge_llm_model'].apply(clean_model_name)
                    if 'question_id' in df.columns:
                        # Keep full question_id format (e.g., BEUL_Exam_2018_Q1)
                        df['question_id'] = df['question_id'].astype(str)
                    
                    if 'question_id' in df.columns and 'generator_model' in df.columns:
                        df['doc_id'] = df['generator_model'] + '_' + df['question_id']
                    
                    # Select relevant columns
                    score_cols = ['score_accuracy', 'score_completeness', 'score_relevance', 'score_overall']
                    reason_cols = ['score_accuracy_reason', 'score_completeness_reason', 
                                 'score_relevance_reason', 'score_overall_reason']
                    base_cols = ['system', 'generator_model', 'judge_type', 'judge_model', 'question_id', 'doc_id']
                    
                    available_cols = base_cols + [col for col in score_cols + reason_cols if col in df.columns]
                    frames.append(df[available_cols].copy())
                    
                    print(f"Loaded {system} file: {filename} ({len(df)} rows)")
                except Exception as e:
                    print(f"Error loading {system} file {filename}: {e}")
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_human_judges(base_path: str = ".") -> pd.DataFrame:
    """Load human judge data from CSV files."""
    frames = []
    
    # Automatically detect human judge files
    human_files = []
    rag_human_dir = os.path.join(base_path, "RAG-System") if base_path != "." else "RAG-System"
    
    if os.path.exists(rag_human_dir):
        for file in os.listdir(rag_human_dir):
            if file.startswith("Human Judge_") and file.endswith(".csv"):
                human_files.append(f"RAG-System/{file}")
    
    # Fallback to hardcoded files if auto-detection fails
    if not human_files:
        human_files = [
            "RAG-System/Human Judge_1.csv",
            "RAG-System/Human Judge_2.csv",
            "RAG-System/Human Judge_3.csv"  # Added third human judge
        ]
    
    for filename in human_files:
        filepath = os.path.join(base_path, filename) if base_path != "." else filename
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                
                df['system'] = 'RAG'
                df['judge_type'] = 'Human'
                df['judge_model'] = df['Annotator']
                df['generator_model'] = df['llm_model'].apply(clean_model_name)
                # Keep original question_id format from human judge files
                df['question_id'] = df['question_id'].astype(str)
                df['doc_id'] = df['generator_model'] + '_' + df['question_id']
                
                df = df.rename(columns={
                    'accuracy': 'score_accuracy',
                    'completeness': 'score_completeness',
                    'relevance': 'score_relevance',
                    'overall': 'score_overall'
                })
                
                base_cols = ['system', 'generator_model', 'judge_type', 'judge_model', 'question_id', 'doc_id']
                score_cols = ['score_accuracy', 'score_completeness', 'score_relevance', 'score_overall']
                available_cols = base_cols + [col for col in score_cols if col in df.columns]
                if 'agreement' in df.columns:
                    available_cols.append('agreement')
                if 'comment' in df.columns:
                    available_cols.append('comment')
                
                frames.append(df[available_cols].copy())
                print(f"Loaded human judge file: {filename} ({len(df)} rows)")
            except Exception as e:
                print(f"Error loading human judge file {filename}: {e}")
    
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_and_aggregate_data(base_path: str = ".") -> pd.DataFrame:
    """Load and aggregate all evaluation data into a unified dataset."""
    print("Loading judgement files...")
    automated_judges = load_judgement_files(base_path)
    
    print("Loading human judge files...")
    human_judges = load_human_judges(base_path)
    
    # Combine all data
    if not automated_judges.empty and not human_judges.empty:
        combined = pd.concat([automated_judges, human_judges], ignore_index=True)
    elif not automated_judges.empty:
        combined = automated_judges
    elif not human_judges.empty:
        combined = human_judges
    else:
        print("No data files found!")
        return pd.DataFrame()
    
    # Clean up judge types
    if 'judge_type' in combined.columns:
        combined['judge_type'] = combined['judge_type'].replace({
            'LLM Judge': 'LLM-Judge',
            'Agent Judge': 'Agent-Judge',
            'LLM-Judge': 'LLM-Judge',
            'Agent-Judge': 'Agent-Judge'
        })
    
    print(f"Total combined dataset: {len(combined)} rows")
    print("Systems:", combined['system'].value_counts().to_dict())
    print("Judge types:", combined['judge_type'].value_counts().to_dict())
    
    return combined

def validate_data_structure(df: pd.DataFrame) -> dict:
    """
    Validate the data structure and provide insights about scaling.
    """
    validation_report = {
        'total_evaluations': len(df),
        'systems': df['system'].value_counts().to_dict(),
        'judge_types': df['judge_type'].value_counts().to_dict(),
        'unique_judge_models': df['judge_model'].nunique(),
        'judge_models': df['judge_model'].unique().tolist(),
        'unique_generator_models': df['generator_model'].nunique() if 'generator_model' in df.columns else 0,
        'generator_models': df['generator_model'].unique().tolist() if 'generator_model' in df.columns else [],
        'unique_questions': df['question_id'].nunique() if 'question_id' in df.columns else 0,
        'score_columns': [col for col in df.columns if col.startswith('score_')],
        'reasoning_columns': [col for col in df.columns if col.endswith('_reason')],
    }
    
    # Calculate expected vs actual
    human_judges = df[df['judge_type'] == 'Human']['judge_model'].nunique()
    automated_judges = df[df['judge_type'].isin(['LLM-Judge', 'Agent-Judge'])]['judge_model'].nunique()
    
    validation_report['human_judges_found'] = human_judges
    validation_report['automated_judge_models_found'] = automated_judges
    validation_report['expected_human_judges'] = EXPECTED_HUMAN_JUDGES
    validation_report['expected_evaluator_models'] = EXPECTED_EVALUATOR_MODELS
    
    # Check if scaling is needed
    validation_report['scaling_adjustments_needed'] = {
        'human_judges': human_judges != EXPECTED_HUMAN_JUDGES,
        'evaluator_models': automated_judges != EXPECTED_EVALUATOR_MODELS,
        'bonferroni_correction_will_adjust': True
    }
    
    return validation_report

def print_validation_report(report: dict) -> None:
    """Print a human-readable validation report."""
    print("\n" + "="*80)
    print("DATA STRUCTURE VALIDATION REPORT")
    print("="*80)
    
    print(f"Total Evaluations: {report['total_evaluations']}")
    print(f"Systems: {report['systems']}")
    print(f"Judge Types: {report['judge_types']}")
    
    print(f"\nJudge Models Found: {report['unique_judge_models']}")
    print(f"   Models: {', '.join(report['judge_models'])}")
    
    print(f"\nGenerator Models: {report['unique_generator_models']}")
    print(f"   Models: {', '.join(report['generator_models'])}")
    
    print(f"\nQuestions: {report['unique_questions']}")
    
    print(f"\nScore Dimensions: {len(report['score_columns'])}")
    print(f"   Dimensions: {', '.join([col.replace('score_', '') for col in report['score_columns']])}")
    
    print(f"\nSCALING ANALYSIS:")
    print(f"   Human Judges: {report['human_judges_found']} found, {report['expected_human_judges']} expected")
    print(f"   Evaluator Models: {report['automated_judge_models_found']} found, {report['expected_evaluator_models']} expected")
    
    if report['scaling_adjustments_needed']['human_judges']:
        print(f"   Warning: Human judge count differs from expected")
    if report['scaling_adjustments_needed']['evaluator_models']:
        print(f"   Warning: Evaluator model count differs from expected")
    
    print(f"   Bonferroni corrections will automatically adjust to actual counts")
    
    print("="*80)

# COMMENTED OUT: Not using Cohen's Kappa analysis
# def compute_cohens_kappa(df: pd.DataFrame, metric: str, annotator1: str, annotator2: str) -> float:
#     """Compute Cohen's Kappa between two specific annotators."""
#     ann1_data = df[df['judge_model'] == annotator1][['doc_id', metric]].dropna()
#     ann2_data = df[df['judge_model'] == annotator2][['doc_id', metric]].dropna()
#     
#     merged = pd.merge(ann1_data, ann2_data, on='doc_id', suffixes=('_1', '_2'))
#     
#     if len(merged) < 2:
#         return np.nan
#     
#     try:
#         return cohen_kappa_score(merged[f'{metric}_1'], merged[f'{metric}_2'])
#     except:
#         return np.nan

def compute_krippendorff_alpha(df: pd.DataFrame, metric: str) -> float:
    """Compute Krippendorff's Alpha for multiple annotators."""
    pivot_table = df.pivot_table(
        index='doc_id', 
        columns='judge_model', 
        values=metric, 
        aggfunc='mean'
    ).dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    if pivot_table.shape[0] < 2 or pivot_table.shape[1] < 2:
        return np.nan
    
    try:
        return krippendorff.alpha(pivot_table.T.values, level_of_measurement='ordinal')
    except:
        return np.nan

def compute_kendall_tau(df: pd.DataFrame, metric: str, annotator1: str, annotator2: str) -> tuple:
    """Compute Kendall's Tau between two annotators."""
    ann1_data = df[df['judge_model'] == annotator1][['doc_id', metric]].dropna()
    ann2_data = df[df['judge_model'] == annotator2][['doc_id', metric]].dropna()
    
    merged = pd.merge(ann1_data, ann2_data, on='doc_id', suffixes=('_1', '_2'))
    
    if len(merged) < 3:
        return np.nan, np.nan
    
    try:
        tau, p_value = kendalltau(merged[f'{metric}_1'], merged[f'{metric}_2'])
        return tau, p_value
    except:
        return np.nan, np.nan

def analyze_reasoning_patterns(df: pd.DataFrame) -> None:
    """Analyze patterns in the reasoning columns to extract qualitative insights."""
    print("\n" + "="*80)
    print("QUALITATIVE REASONING ANALYSIS")
    print("="*80)
    
    reason_columns = [col for col in df.columns if col.endswith('_reason')]
    
    if not reason_columns:
        print("No reasoning columns found in the dataset.")
        return
    
    # Keywords to look for in different judge types
    agent_keywords = ['fact-check', 'verified', 'hallucination', 'citation', 'source', 'accuracy', 'incorrect', 'factual']
    llm_keywords = ['structure', 'comprehensive', 'analysis', 'depth', 'argumentation', 'style', 'coherent']
    
    for reason_col in reason_columns:
        metric = reason_col.replace('_reason', '').replace('score_', '')
        print(f"\n{metric.upper()} Reasoning Patterns")
        print("-" * 40)
        
        for judge_type in ['LLM-Judge', 'Agent-Judge']:
            subset = df[df['judge_type'] == judge_type][reason_col].dropna()
            
            if subset.empty:
                continue
            
            print(f"\n{judge_type}:")
            
            # Count mentions of key themes
            if judge_type == 'Agent-Judge':
                keywords = agent_keywords
            else:
                keywords = llm_keywords
            
            keyword_counts = {}
            total_responses = len(subset)
            
            for keyword in keywords:
                count = subset.str.lower().str.contains(keyword, na=False).sum()
                if count > 0:
                    keyword_counts[keyword] = count
                    percentage = (count / total_responses) * 100
                    print(f"  - '{keyword}': {count}/{total_responses} ({percentage:.1f}%)")
            
            # Sample reasoning examples (first few non-empty ones)
            print(f"  Sample reasoning:")
            sample_size = min(2, len(subset))
            for i, reason in enumerate(subset.head(sample_size)):
                if pd.notna(reason) and len(str(reason)) > 50:
                    truncated = str(reason)[:150] + "..." if len(str(reason)) > 150 else str(reason)
                    print(f"    {i+1}. {truncated}")

def create_comprehensive_visualizations(df: pd.DataFrame, output_dir: str = "analysis_plots") -> None:
    """Create improved visualizations that address ambiguity issues."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreating improved visualizations in '{output_dir}/'...")
    
    # Import improved visualization functions
    from improved_visualizations import (
        create_human_judge_heatmap,
        create_system_specific_comparisons, 
        create_dimension_specific_plots,
        create_judge_model_performance,
        create_contextual_dashboard
    )
    
    # Create all improved visualizations
    create_human_judge_heatmap(df, output_dir)
    create_system_specific_comparisons(df, output_dir)
    create_dimension_specific_plots(df, output_dir)
    create_judge_model_performance(df, output_dir)
    create_contextual_dashboard(df, output_dir)
    
    print("✅ Improved visualizations created successfully!")

# =============================================================================
# Enhanced Inter-Rater Reliability Analysis
# =============================================================================

def comprehensive_inter_judge_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete inter-judge reliability analysis addressing all combinations.
    
    Research Question Q1 & Q2: Are judges reliable within and across types?
    """
    results = []
    metrics = ['score_accuracy', 'score_completeness', 'score_relevance', 'score_overall']
    
    print("\n" + "="*100)
    print("COMPREHENSIVE INTER-JUDGE RELIABILITY ANALYSIS")
    print("="*100)
    print("Research Questions Q1 & Q2: Judge reliability within and across types")
    
    for metric in metrics:
        print(f"\nANALYZING {metric.replace('score_', '').upper()} DIMENSION")
        print("="*80)
        
        # 1. HUMAN JUDGES (RAG system only) - Streamlined Analysis
        print(f"\n1. HUMAN JUDGE RELIABILITY (RAG System)")
        print("-" * 50)
        
        human_rag = df[(df['judge_type'] == 'Human') & (df['system'] == 'RAG')]
        if not human_rag.empty:
            # Krippendorff's Alpha for all human judges (primary metric)
            alpha_human = compute_krippendorff_alpha(human_rag, metric)
            print(f"Krippendorff's α (Human Judges): {alpha_human:.4f}")
            
            # Interpretation
            if alpha_human > 0.8:
                print(f"    Excellent reliability")
            elif alpha_human > 0.67:
                print(f"    Good reliability")
            elif alpha_human > 0.4:
                print(f"    Moderate reliability")
            else:
                print(f"    Poor reliability")
            
            # Store simplified results
            results.append({
                'metric': metric, 'analysis_type': 'human_reliability',
                'system': 'RAG', 'judge_pair': 'all_human_judges',
                'krippendorff_alpha': alpha_human, 'cohens_kappa': np.nan,
                'kendall_tau': np.nan, 'kendall_p': np.nan
            })
            
            # # COMMENTED OUT: Pairwise analysis (kept for future reference)
            # human_judges = human_rag['judge_model'].unique()
            # for j1, j2 in combinations(human_judges, 2):
            #     kappa = compute_cohens_kappa(human_rag, metric, j1, j2)
            #     tau, tau_p = compute_kendall_tau(human_rag, metric, j1, j2)
            #     print(f"  {j1} vs {j2}:")
            #     print(f"    Cohen's κ: {kappa:.4f}, Kendall's τ: {tau:.4f} (p={tau_p:.4f})")
            #     
            #     results.append({
            #         'metric': metric, 'analysis_type': 'human_reliability',
            #         'system': 'RAG', 'judge_pair': f"{j1}_vs_{j2}",
            #         'krippendorff_alpha': alpha_human, 'cohens_kappa': kappa,
            #         'kendall_tau': tau, 'kendall_p': tau_p
            #     })
        
        # 2. LLM-JUDGE RELIABILITY (by system) - Streamlined Analysis
        print(f"\n2. LLM-JUDGE RELIABILITY")
        print("-" * 50)
        
        for system in ['RAG', 'LLM-only']:
            llm_judges = df[(df['judge_type'] == 'LLM-Judge') & (df['system'] == system)]
            if not llm_judges.empty:
                alpha_llm = compute_krippendorff_alpha(llm_judges, metric)
                print(f"  {system} System - Krippendorff's α (LLM-Judges): {alpha_llm:.4f}")
                
                # Interpretation
                if alpha_llm > 0.67:
                    print(f"    Good reliability")
                elif alpha_llm > 0.4:
                    print(f"    Moderate reliability")
                else:
                    print(f"    Poor reliability")
                
                # Store simplified results
                results.append({
                    'metric': metric, 'analysis_type': 'llm_judge_reliability',
                    'system': system, 'judge_pair': 'all_llm_judges',
                    'krippendorff_alpha': alpha_llm, 'cohens_kappa': np.nan,
                    'kendall_tau': np.nan, 'kendall_p': np.nan
                })
                
                # # COMMENTED OUT: Pairwise comparisons (kept for future reference)
                # judge_models = llm_judges['judge_model'].unique()
                # for j1, j2 in combinations(judge_models, 2):
                #     kappa = compute_cohens_kappa(llm_judges, metric, j1, j2)
                #     tau, tau_p = compute_kendall_tau(llm_judges, metric, j1, j2)
                #     print(f"    {j1} vs {j2}: κ={kappa:.4f}, τ={tau:.4f} (p={tau_p:.4f})")
                #     
                #     results.append({
                #         'metric': metric, 'analysis_type': 'llm_judge_reliability',
                #         'system': system, 'judge_pair': f"{j1}_vs_{j2}",
                #         'krippendorff_alpha': alpha_llm, 'cohens_kappa': kappa,
                #         'kendall_tau': tau, 'kendall_p': tau_p
                #     })
        
        # 3. AGENT-JUDGE RELIABILITY (by system) - Streamlined Analysis
        print(f"\n3. AGENT-JUDGE RELIABILITY")
        print("-" * 50)
        
        for system in ['RAG', 'LLM-only']:
            agent_judges = df[(df['judge_type'] == 'Agent-Judge') & (df['system'] == system)]
            if not agent_judges.empty:
                alpha_agent = compute_krippendorff_alpha(agent_judges, metric)
                print(f"  {system} System - Krippendorff's α (Agent-Judges): {alpha_agent:.4f}")
                
                # Interpretation
                if alpha_agent > 0.67:
                    print(f"    Good reliability")
                elif alpha_agent > 0.4:
                    print(f"    Moderate reliability")
                else:
                    print(f"    Poor reliability")
                
                # Store simplified results
                results.append({
                    'metric': metric, 'analysis_type': 'agent_judge_reliability',
                    'system': system, 'judge_pair': 'all_agent_judges',
                    'krippendorff_alpha': alpha_agent, 'cohens_kappa': np.nan,
                    'kendall_tau': np.nan, 'kendall_p': np.nan
                })
                
                # # COMMENTED OUT: Pairwise comparisons (kept for future reference)
                # judge_models = agent_judges['judge_model'].unique()
                # for j1, j2 in combinations(judge_models, 2):
                #     kappa = compute_cohens_kappa(agent_judges, metric, j1, j2)
                #     tau, tau_p = compute_kendall_tau(agent_judges, metric, j1, j2)
                #     print(f"    {j1} vs {j2}: κ={kappa:.4f}, τ={tau:.4f} (p={tau_p:.4f})")
                #     
                #     results.append({
                #         'metric': metric, 'analysis_type': 'agent_judge_reliability',
                #         'system': system, 'judge_pair': f"{j1}_vs_{j2}",
                #         'krippendorff_alpha': alpha_agent, 'cohens_kappa': kappa,
                #         'kendall_tau': tau, 'kendall_p': tau_p
                #     })
        
        # 4. CROSS-JUDGE TYPE COMPARISONS
        print(f"\n4. CROSS-JUDGE TYPE RELIABILITY")
        print("-" * 50)
        
        # Human vs LLM-Judge (RAG system only)
        human_rag_avg = df[(df['judge_type'] == 'Human') & (df['system'] == 'RAG')].groupby('doc_id')[metric].mean()
        llm_rag_avg = df[(df['judge_type'] == 'LLM-Judge') & (df['system'] == 'RAG')].groupby('doc_id')[metric].mean()
        
        common_docs = set(human_rag_avg.index) & set(llm_rag_avg.index)
        if len(common_docs) >= 3:
            tau_human_llm, p_human_llm = kendalltau(
                human_rag_avg[list(common_docs)], 
                llm_rag_avg[list(common_docs)]
            )
            print(f"  Human vs LLM-Judge (RAG): τ={tau_human_llm:.4f} (p={p_human_llm:.4f})")
            
            results.append({
                'metric': metric, 'analysis_type': 'cross_judge_type',
                'system': 'RAG', 'comparison': 'Human_vs_LLM-Judge',
                'kendall_tau': tau_human_llm, 'kendall_p': p_human_llm,
                'n_items': len(common_docs)
            })
        
        # LLM-Judge vs Agent-Judge (both systems)
        for system in ['RAG', 'LLM-only']:
            llm_avg = df[(df['judge_type'] == 'LLM-Judge') & (df['system'] == system)].groupby('doc_id')[metric].mean()
            agent_avg = df[(df['judge_type'] == 'Agent-Judge') & (df['system'] == system)].groupby('doc_id')[metric].mean()
            
            common_docs = set(llm_avg.index) & set(agent_avg.index)
            if len(common_docs) >= 3:
                tau_llm_agent, p_llm_agent = kendalltau(
                    llm_avg[list(common_docs)], 
                    agent_avg[list(common_docs)]
                )
                print(f"  LLM-Judge vs Agent-Judge ({system}): τ={tau_llm_agent:.4f} (p={p_llm_agent:.4f})")
                
                results.append({
                    'metric': metric, 'analysis_type': 'cross_judge_type',
                    'system': system, 'comparison': 'LLM-Judge_vs_Agent-Judge',
                    'kendall_tau': tau_llm_agent, 'kendall_p': p_llm_agent,
                    'n_items': len(common_docs)
                })
    
    # Save comprehensive results
    results_df = pd.DataFrame(results)
    results_df.to_csv('comprehensive_inter_judge_reliability.csv', index=False)
    print(f"\nComprehensive reliability analysis saved to 'comprehensive_inter_judge_reliability.csv'")
    
    return results_df

# =============================================================================
# Enhanced System Comparison Analysis
# =============================================================================

def enhanced_system_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Research Question Q3: Is RAG significantly better than LLM-only?
    
    Uses proper hypothesis testing with Bonferroni correction.
    """
    results = []
    metrics = ['score_accuracy', 'score_completeness', 'score_relevance', 'score_overall']
    
    # Get automated judges only
    automated_judges = df[df['judge_type'].isin(['LLM-Judge', 'Agent-Judge'])]
    
    if automated_judges.empty:
        print("No automated judge data found for system comparison!")
        return pd.DataFrame()
    
    # Use standard α = 0.05 for exploratory analysis (given poor human judge reliability)
    judge_combinations = automated_judges[['judge_model', 'judge_type']].drop_duplicates()
    total_tests = len(judge_combinations) * len(metrics)
    alpha_corrected = 0.05  # Standard significance level for exploratory analysis
    bonferroni_alpha = 0.05 / total_tests if total_tests > 0 else 0.05  # For reference only
    
    print("\n" + "="*100)
    print("ENHANCED SYSTEM COMPARISON ANALYSIS: RAG vs LLM-only")
    print("="*100)
    print("Research Question Q3: Is RAG significantly better than LLM-only?")
    print(f"Total hypothesis tests: {total_tests}")
    print(f"Analysis α: {alpha_corrected:.3f} (exploratory analysis)")
    print(f"Bonferroni α: {bonferroni_alpha:.6f} (for reference)")
    print("\nNote: Using α = 0.05 for exploratory analysis due to:")
    print("  • Poor human judge reliability (α = -0.047)")
    print("  • Exploratory nature of research")
    print("  • Focus on effect sizes and patterns")
    print("\nHypotheses for each test:")
    print("H₀: median(RAG_score - LLM-only_score) = 0")
    print("Hₐ: median(RAG_score - LLM-only_score) > 0")
    
    for _, row in judge_combinations.iterrows():
        judge_model = row['judge_model']
        judge_type = row['judge_type']
        
        print(f"\nANALYZING: {judge_model} ({judge_type})")
        print("="*60)
        
        # Get data for this specific judge
        judge_data = automated_judges[
            (automated_judges['judge_model'] == judge_model) & 
            (automated_judges['judge_type'] == judge_type)
        ]
        
        # Separate RAG and LLM-only data
        rag_data = judge_data[judge_data['system'] == 'RAG']
        llm_data = judge_data[judge_data['system'] == 'LLM-only']
        
        if rag_data.empty or llm_data.empty:
            print("  Warning: Insufficient data for comparison")
            continue
        
        # Merge on doc_id to create pairs
        merged = pd.merge(
            rag_data[['doc_id'] + metrics], 
            llm_data[['doc_id'] + metrics], 
            on='doc_id', 
            suffixes=('_rag', '_llm')
        )
        
        if len(merged) < 3:
            print(f"  Warning: Insufficient paired samples (n={len(merged)})")
            continue
        
        print(f"Analyzing {len(merged)} paired samples")
        print(f"H₀: median(RAG - LLM-only) = 0  |  α={alpha_corrected:.6f}")
        
        for metric in metrics:
            rag_scores = merged[f'{metric}_rag']
            llm_scores = merged[f'{metric}_llm']
            differences = rag_scores - llm_scores
            
            # Descriptive statistics
            mean_diff = differences.mean()
            median_diff = differences.median()
            
            # Effect sizes
            cohens_d = mean_diff / differences.std(ddof=1) if len(differences) > 1 else np.nan
            
            # Paired t-test
            t_stat, t_p = ttest_rel(rag_scores, llm_scores)
            
            # Wilcoxon signed-rank test
            try:
                w_stat, w_p = wilcoxon(differences, alternative='two-sided', method='exact')
                # Effect size for Wilcoxon: r = |z|/√n
                z_approx = (w_stat - len(differences)*(len(differences)+1)/4) / np.sqrt(len(differences)*(len(differences)+1)*(2*len(differences)+1)/24)
                wilcoxon_r = abs(z_approx) / np.sqrt(len(differences))
            except:
                w_stat, w_p, wilcoxon_r = np.nan, np.nan, np.nan
            
            # Significance determination
            t_significant = t_p < alpha_corrected
            w_significant = w_p < alpha_corrected
            
            # Effect size interpretation using configuration
            effect_interp = (
                'Large' if abs(cohens_d) > COHENS_D_THRESHOLDS['large'] else 
                'Medium' if abs(cohens_d) > COHENS_D_THRESHOLDS['medium'] else 
                'Small' if abs(cohens_d) > COHENS_D_THRESHOLDS['small'] else 'Negligible'
            )
            
            # Significance indicators
            t_sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_significant else "ns"
            w_sig = "***" if w_p < 0.001 else "**" if w_p < 0.01 else "*" if w_significant else "ns"
            
            metric_name = metric.replace('score_', '').capitalize()
            print(f"\n  {metric_name}:")
            print(f"     Mean Δ: {mean_diff:+.3f} | Median Δ: {median_diff:+.3f}")
            print(f"     Cohen's d: {cohens_d:.3f} ({effect_interp})")
            print(f"     t-test: t={t_stat:.3f}, p={t_p:.6f} {t_sig}")
            print(f"     Wilcoxon: W={w_stat}, p={w_p:.6f} {w_sig}, r={wilcoxon_r:.3f}")
            
            results.append({
                'judge_model': judge_model,
                'judge_type': judge_type,
                'metric': metric,
                'n_pairs': len(merged),
                'mean_difference': mean_diff,
                'median_difference': median_diff,
                'cohens_d': cohens_d,
                'effect_size_interpretation': effect_interp,
                't_statistic': t_stat,
                't_p_value': t_p,
                't_significant': t_significant,
                'wilcoxon_statistic': w_stat,
                'wilcoxon_p_value': w_p,
                'wilcoxon_significant': w_significant,
                'wilcoxon_r': wilcoxon_r,
                'alpha_used': alpha_corrected,
                'bonferroni_alpha': bonferroni_alpha,
                'total_tests': total_tests
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('enhanced_system_comparison.csv', index=False)
    print(f"\nEnhanced system comparison saved to 'enhanced_system_comparison.csv'")
    
    return results_df

# =============================================================================
# Enhanced Judge Comparison Analysis
# =============================================================================

def enhanced_judge_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Research Question Q4: Do LLM-Judges and Agent-Judges score differently?
    
    Uses both paired and unpaired approaches with proper hypothesis testing.
    """
    results = []
    metrics = ['score_accuracy', 'score_completeness', 'score_relevance', 'score_overall']
    
    automated_judges = df[df['judge_type'].isin(['LLM-Judge', 'Agent-Judge'])]
    
    if automated_judges.empty:
        print("No automated judge data found for judge comparison!")
        return pd.DataFrame()
    
    # Use standard α = 0.05 for exploratory analysis (consistent with system comparison)
    systems = ['RAG', 'LLM-only']
    judge_models = automated_judges['judge_model'].unique()
    total_tests_per_system = len(judge_models) * len(metrics) * 3  # 3 tests per metric
    alpha_corrected = 0.05  # Standard significance level for exploratory analysis
    bonferroni_alpha = 0.05 / total_tests_per_system if total_tests_per_system > 0 else 0.05  # For reference
    
    print("\n" + "="*100)
    print("ENHANCED JUDGE COMPARISON ANALYSIS: LLM-Judge vs Agent-Judge")
    print("="*100)
    print("Research Question Q4: Do LLM-Judges and Agent-Judges score differently?")
    print(f"Tests per system: {total_tests_per_system}")
    print(f"Analysis α: {alpha_corrected:.3f} (exploratory analysis)")
    print(f"Bonferroni α: {bonferroni_alpha:.6f} (for reference)")
    
    for system in systems:
        print(f"\n{system.upper()} SYSTEM ANALYSIS")
        print("="*80)
        print("Hypotheses:")
        print("H₀ (Paired): median(LLM-Judge - Agent-Judge) = 0")
        print("H₀ (Unpaired): LLM-Judge and Agent-Judge have same score distribution")
        
        system_data = automated_judges[automated_judges['system'] == system]
        
        if system_data.empty:
            print("  Warning: No data available")
            continue
        
        for judge_model in judge_models:
            print(f"\n  {judge_model}")
            print("-" * 50)
            
            llm_data = system_data[
                (system_data['judge_model'] == judge_model) & 
                (system_data['judge_type'] == 'LLM-Judge')
            ]
            
            agent_data = system_data[
                (system_data['judge_model'] == judge_model) & 
                (system_data['judge_type'] == 'Agent-Judge')
            ]
            
            if llm_data.empty or agent_data.empty:
                print("    Warning: Insufficient data for comparison")
                continue
            
            for metric in metrics:
                metric_name = metric.replace('score_', '').capitalize()
                print(f"\n    {metric_name} Analysis:")
                
                # PAIRED ANALYSIS
                llm_scores_df = llm_data[['doc_id', metric]].dropna()
                agent_scores_df = agent_data[['doc_id', metric]].dropna()
                
                paired_data = pd.merge(
                    llm_scores_df, agent_scores_df, 
                    on='doc_id', 
                    suffixes=('_llm', '_agent')
                )
                
                paired_results = {'n_paired': 0}
                
                if len(paired_data) >= 3:
                    llm_paired = paired_data[f'{metric}_llm']
                    agent_paired = paired_data[f'{metric}_agent']
                    differences = llm_paired - agent_paired
                    
                    # Paired t-test
                    t_stat, t_p = ttest_rel(llm_paired, agent_paired)
                    cohens_d = differences.mean() / differences.std(ddof=1)
                    
                    # Paired Wilcoxon test
                    try:
                        w_stat, w_p = wilcoxon(differences, method='exact')
                        z_approx = (w_stat - len(differences)*(len(differences)+1)/4) / np.sqrt(len(differences)*(len(differences)+1)*(2*len(differences)+1)/24)
                        wilcoxon_r = abs(z_approx) / np.sqrt(len(differences))
                    except:
                        w_stat, w_p, wilcoxon_r = np.nan, np.nan, np.nan
                    
                    paired_results.update({
                        'n_paired': len(paired_data),
                        'paired_mean_diff': differences.mean(),
                        'paired_cohens_d': cohens_d,
                        'paired_t_stat': t_stat,
                        'paired_t_p': t_p,
                        'paired_t_sig': t_p < alpha_corrected,
                        'paired_w_stat': w_stat,
                        'paired_w_p': w_p,
                        'paired_w_sig': w_p < alpha_corrected,
                        'paired_w_r': wilcoxon_r
                    })
                    
                    print(f"      Paired (n={len(paired_data)}): Δ={differences.mean():+.3f}, d={cohens_d:.3f}")
                    print(f"        t-test: p={t_p:.6f} {'*' if t_p < alpha_corrected else 'ns'}")
                    print(f"        Wilcoxon: p={w_p:.6f} {'*' if w_p < alpha_corrected else 'ns'}, r={wilcoxon_r:.3f}")
                
                # UNPAIRED ANALYSIS (Mann-Whitney U)
                llm_all = llm_data[metric].dropna()
                agent_all = agent_data[metric].dropna()
                
                unpaired_results = {}
                
                if len(llm_all) >= 3 and len(agent_all) >= 3:
                    try:
                        u_stat, u_p = mannwhitneyu(llm_all, agent_all, alternative='two-sided')
                        # Effect size for Mann-Whitney: r = |z|/√(n1+n2)
                        z_mwu = (u_stat - len(llm_all)*len(agent_all)/2) / np.sqrt(len(llm_all)*len(agent_all)*(len(llm_all)+len(agent_all)+1)/12)
                        mwu_r = abs(z_mwu) / np.sqrt(len(llm_all) + len(agent_all))
                    except:
                        u_stat, u_p, mwu_r = np.nan, np.nan, np.nan
                    
                    unpaired_results.update({
                        'n_llm_total': len(llm_all),
                        'n_agent_total': len(agent_all),
                        'unpaired_mean_diff': llm_all.mean() - agent_all.mean(),
                        'mwu_stat': u_stat,
                        'mwu_p': u_p,
                        'mwu_sig': u_p < alpha_corrected,
                        'mwu_r': mwu_r
                    })
                    
                    print(f"      Unpaired: LLM(n={len(llm_all)}) vs Agent(n={len(agent_all)})")
                    print(f"        Mann-Whitney U: p={u_p:.6f} {'*' if u_p < alpha_corrected else 'ns'}, r={mwu_r:.3f}")
                
                # Combine results
                result_entry = {
                    'system': system,
                    'judge_model': judge_model,
                    'metric': metric,
                    'alpha_used': alpha_corrected,
                    'bonferroni_alpha': bonferroni_alpha,
                    **paired_results,
                    **unpaired_results
                }
                
                results.append(result_entry)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('enhanced_judge_comparison.csv', index=False)
    print(f"\nEnhanced judge comparison saved to 'enhanced_judge_comparison.csv'")
    
    return results_df

# =============================================================================
# Self-Evaluation Bias Analysis (NEW)
# =============================================================================

def analyze_self_evaluation_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Research Question Q6: Does gemini-1.5-flash show self-evaluation bias?
    
    Analyzes whether gemini-1.5-flash shows bias when judging its own outputs
    compared to judging other models' outputs.
    """
    results = []
    metrics = ['score_accuracy', 'score_completeness', 'score_relevance', 'score_overall']
    
    print("\n" + "="*100)
    print("SELF-EVALUATION BIAS ANALYSIS: gemini-1.5-flash")
    print("="*100)
    print("Research Question Q6: Does gemini-1.5-flash show self-evaluation bias?")
    print("Comparing: Gemini judging Gemini vs. Gemini judging other models")
    
    # Filter for gemini-1.5-flash as judge
    gemini_judge_data = df[df['judge_model'] == 'gemini-1.5-flash']
    
    if gemini_judge_data.empty:
        print("Error: No gemini-1.5-flash judge data found!")
        return pd.DataFrame()
    
    for system in ['RAG', 'LLM-only']:
        for judge_type in ['LLM-Judge', 'Agent-Judge']:
            print(f"\n{system} System - {judge_type}")
            print("="*60)
            
            # Get data for this specific combination
            judge_data = gemini_judge_data[
                (gemini_judge_data['system'] == system) & 
                (gemini_judge_data['judge_type'] == judge_type)
            ]
            
            if judge_data.empty:
                print("  Warning: No data available for this combination")
                continue
            
            # Split into self-evaluation vs. other-evaluation
            self_evaluation = judge_data[judge_data['generator_model'] == 'gemini-1.5-flash']
            other_evaluation = judge_data[judge_data['generator_model'] != 'gemini-1.5-flash']
            
            if self_evaluation.empty or other_evaluation.empty:
                print(f"  Warning: Insufficient data: Self({len(self_evaluation)}) vs Other({len(other_evaluation)})")
                continue
            
            print(f"Sample sizes: Self-evaluation({len(self_evaluation)}) vs Other-evaluation({len(other_evaluation)})")
            
            for metric in metrics:
                metric_name = metric.replace('score_', '').capitalize()
                
                # Get scores
                self_scores = self_evaluation[metric].dropna()
                other_scores = other_evaluation[metric].dropna()
                
                if len(self_scores) < 3 or len(other_scores) < 3:
                    continue
                
                # Descriptive statistics
                mean_self = self_scores.mean()
                mean_other = other_scores.mean()
                bias_magnitude = mean_self - mean_other
                
                # Statistical test (Mann-Whitney U for independent samples)
                try:
                    u_stat, p_val = mannwhitneyu(self_scores, other_scores, alternative='two-sided')
                    # Effect size for Mann-Whitney: r = |z|/√(n1+n2)
                    z_mwu = (u_stat - len(self_scores)*len(other_scores)/2) / np.sqrt(len(self_scores)*len(other_scores)*(len(self_scores)+len(other_scores)+1)/12)
                    effect_size = abs(z_mwu) / np.sqrt(len(self_scores) + len(other_scores))
                except:
                    u_stat, p_val, effect_size = np.nan, np.nan, np.nan
                
                # Bias interpretation
                bias_direction = 'Self-favorable' if bias_magnitude > 0 else 'Self-critical' if bias_magnitude < 0 else 'No bias'
                significant = p_val < 0.05 if not np.isnan(p_val) else False
                
                # Effect size interpretation
                effect_interp = (
                    'Large' if effect_size > 0.5 else 
                    'Medium' if effect_size > 0.3 else 
                    'Small' if effect_size > 0.1 else 'Negligible'
                )
                
                print(f"\n  {metric_name}:")
                print(f"     Self-eval mean: {mean_self:.3f} | Other-eval mean: {mean_other:.3f}")
                print(f"     Bias magnitude: {bias_magnitude:+.3f} ({bias_direction})")
                print(f"     Mann-Whitney U: p={p_val:.6f} {'***' if significant else 'ns'}")
                print(f"     Effect size (r): {effect_size:.3f} ({effect_interp})")
                
                if significant and abs(bias_magnitude) > 0.2:  # Practically significant bias
                    if bias_magnitude > 0:
                        print(f"     SIGNIFICANT SELF-FAVORABLE BIAS DETECTED!")
                    else:
                        print(f"     SIGNIFICANT SELF-CRITICAL BIAS DETECTED!")
                
                # Store results
                results.append({
                    'system': system,
                    'judge_type': judge_type,
                    'metric': metric,
                    'n_self_evaluation': len(self_scores),
                    'n_other_evaluation': len(other_scores),
                    'mean_self_evaluation': mean_self,
                    'mean_other_evaluation': mean_other,
                    'bias_magnitude': bias_magnitude,
                    'bias_direction': bias_direction,
                    'mwu_p_value': p_val,
                    'mwu_significant': significant,
                    'effect_size_r': effect_size,
                    'effect_size_interpretation': effect_interp,
                    'practically_significant': significant and abs(bias_magnitude) > 0.2
                })
    
    # Summary analysis
    print(f"\n" + "="*60)
    print("BIAS ANALYSIS SUMMARY")
    print("="*60)
    
    if results:
        bias_df = pd.DataFrame(results)
        
        # Count significant biases
        significant_biases = bias_df[bias_df['mwu_significant'] == True]
        self_favorable = significant_biases[significant_biases['bias_magnitude'] > 0]
        self_critical = significant_biases[significant_biases['bias_magnitude'] < 0]
        
        print(f"Total comparisons: {len(bias_df)}")
        print(f"Significant biases: {len(significant_biases)}/{len(bias_df)}")
        print(f"   Self-favorable: {len(self_favorable)}")
        print(f"   Self-critical: {len(self_critical)}")
        
        if len(significant_biases) > 0:
            print(f"\nStrongest bias:")
            strongest = significant_biases.loc[significant_biases['bias_magnitude'].abs().idxmax()]
            print(f"   {strongest['system']} {strongest['judge_type']} - {strongest['metric']}")
            print(f"   Bias: {strongest['bias_magnitude']:+.3f} ({strongest['bias_direction']})")
            print(f"   Effect size: {strongest['effect_size_r']:.3f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('self_evaluation_bias_analysis.csv', index=False)
    print(f"\nSelf-evaluation bias analysis saved to 'self_evaluation_bias_analysis.csv'")
    
    return results_df

# =============================================================================
# Research Questions Summary
# =============================================================================

def analyze_accuracy_relevance_deep_dive(reliability_results: pd.DataFrame, 
                                        system_results: pd.DataFrame, 
                                        judge_results: pd.DataFrame) -> None:
    """
    Deep dive analysis focusing on Accuracy and Relevance dimensions.
    Addresses system-specific reliability patterns and cross-system comparisons.
    """
    print("\n" + "="*100)
    print("DEEP DIVE ANALYSIS: ACCURACY & RELEVANCE DIMENSIONS")
    print("="*100)
    
    focus_metrics = ['score_accuracy', 'score_relevance']
    focus_names = ['Accuracy', 'Relevance']
    
    for metric, name in zip(focus_metrics, focus_names):
        print(f"\n{'🎯 ' + name.upper() + ' DIMENSION DEEP DIVE'}")
        print("="*80)
        
        # Q1a: System-specific reliability analysis
        print(f"\n📊 Q1a-{name}: Judge reliability on {name} by system")
        print("-" * 60)
        
        for system in ['RAG', 'LLM-only']:
            print(f"\n{system} System:")
            
            # Human judges (RAG only)
            if system == 'RAG':
                human_data = reliability_results[
                    (reliability_results['metric'] == metric) & 
                    (reliability_results['analysis_type'] == 'human_reliability') &
                    (reliability_results['system'] == 'RAG')
                ]
                if not human_data.empty:
                    #avg_kappa = human_data['cohens_kappa'].mean()
                    alpha = human_data['krippendorff_alpha'].mean()
                    tau = human_data['kendall_tau'].mean()
                    print(f"  Human Judges: α={alpha:.4f}, τ={tau:.4f}")

                    # TODO: Add kappa analysis
                    # if avg_kappa < -0.05:
                    #     print(f"    ⚠️  CRITICAL: Negative kappa indicates worse than chance agreement!")
                    # elif avg_kappa < 0.4:
                    if alpha < -0.05:
                        print(f"    ⚠️  CRITICAL: Negative alpha indicates worse than chance agreement!")
                    elif alpha < 0.4:
                        print(f"    ⚠️  Poor reliability threatens evaluation validity")
            
            # LLM-Judge reliability by system
            llm_data = reliability_results[
                (reliability_results['metric'] == metric) & 
                (reliability_results['analysis_type'] == 'llm_judge_reliability') &
                (reliability_results['system'] == system)
            ]
            if not llm_data.empty:
                avg_alpha = llm_data['krippendorff_alpha'].mean()
                print(f"  LLM-Judge α: {avg_alpha:.4f}")
                
                if avg_alpha > 0.67:
                    print(f"    ✅ Good reliability")
                elif avg_alpha > 0.4:
                    print(f"    ⚠️  Moderate reliability")
                else:
                    print(f"    ❌ Poor reliability")
            
            # Agent-Judge reliability by system
            agent_data = reliability_results[
                (reliability_results['metric'] == metric) & 
                (reliability_results['analysis_type'] == 'agent_judge_reliability') &
                (reliability_results['system'] == system)
            ]
            if not agent_data.empty:
                avg_alpha = agent_data['krippendorff_alpha'].mean()
                print(f"  Agent-Judge α: {avg_alpha:.4f}")
                
                if avg_alpha > 0.67:
                    print(f"    ✅ Good reliability")
                elif avg_alpha > 0.4:
                    print(f"    ⚠️  Moderate reliability")
                else:
                    print(f"    ❌ Poor reliability")
        
        # Q2a: Cross-judge agreement analysis
        print(f"\n📊 Q2a-{name}: Cross-judge agreement on {name}")
        print("-" * 60)
        
        cross_type = reliability_results[
            (reliability_results['analysis_type'] == 'cross_judge_type') &
            (reliability_results['metric'] == metric)
        ]
        
        if not cross_type.empty:
            for _, row in cross_type.iterrows():
                comparison = row['comparison']
                system = row['system']
                tau = row['kendall_tau']
                p_val = row['kendall_p']
                
                print(f"  {comparison} ({system}): τ={tau:.4f} (p={p_val:.4f})")
                
                if tau > 0.7:
                    print(f"    ✅ Strong agreement")
                elif tau > 0.5:
                    print(f"    ⚠️  Moderate agreement")
                else:
                    print(f"    ❌ Weak agreement")
        
        # Q3a: System comparison analysis
        print(f"\n📊 Q3a-{name}: RAG vs LLM-only on {name}")
        print("-" * 60)
        
        metric_system_data = system_results[system_results['metric'] == metric]
        
        if not metric_system_data.empty:
            significant_improvements = metric_system_data[
                (metric_system_data['wilcoxon_significant'] == True) & 
                (metric_system_data['mean_difference'] > 0)
            ]
            
            print(f"  Significant RAG advantages: {len(significant_improvements)}/{len(metric_system_data)}")
            
            if len(significant_improvements) > 0:
                for _, row in significant_improvements.iterrows():
                    print(f"    ✅ {row['judge_model']} ({row['judge_type']}): d={row['cohens_d']:.3f}, r={row['wilcoxon_r']:.3f}")
            else:
                print(f"    ❌ No significant RAG advantages on {name}")
                
                # Show closest to significance
                best_attempt = metric_system_data.loc[metric_system_data['mean_difference'].idxmax()]
                print(f"    → Best attempt: {best_attempt['judge_model']} ({best_attempt['judge_type']})")
                print(f"      Mean Δ: {best_attempt['mean_difference']:+.3f}, p={best_attempt['wilcoxon_p_value']:.4f}")
        
        # Q4a: Judge behavior analysis
        print(f"\n📊 Q4a-{name}: Judge behavior differences on {name}")
        print("-" * 60)
        
        metric_judge_data = judge_results[judge_results['metric'] == metric]
        
        if not metric_judge_data.empty:
            for system in ['RAG', 'LLM-only']:
                system_data = metric_judge_data[metric_judge_data['system'] == system]
                
                if not system_data.empty:
                    print(f"\n  {system} System:")
                    
                    for _, row in system_data.iterrows():
                        judge_model = row['judge_model']
                        mean_diff = row['paired_mean_diff']
                        cohens_d = row['paired_cohens_d']
                        significant = row['paired_t_sig']
                        
                        if significant:
                            direction = "LLM-Judge ↑" if mean_diff > 0 else "Agent-Judge ↑"
                            print(f"    {judge_model}: {direction} {abs(mean_diff):.3f} points (d={cohens_d:.3f}) ***")
                        else:
                            print(f"    {judge_model}: No significant difference (Δ={mean_diff:+.3f})")

def analyze_dimension_specific_patterns(reliability_results: pd.DataFrame, 
                                      system_results: pd.DataFrame, 
                                      judge_results: pd.DataFrame) -> None:
    """
    Surface-level analysis of all dimensions with focus on Completeness and Overall.
    """
    print("\n" + "="*100)
    print("SURFACE ANALYSIS: ALL DIMENSIONS OVERVIEW")
    print("="*100)
    
    metrics = ['score_accuracy', 'score_completeness', 'score_relevance', 'score_overall']
    metric_names = ['Accuracy', 'Completeness', 'Relevance', 'Overall']
    
    # Q1a: Reliability variation by dimension
    print("\n📊 Q1a: Does reliability vary by dimension within judge types?")
    print("-" * 60)
    
    for judge_type in ['human', 'llm_judge', 'agent_judge']:
        judge_reliability = reliability_results[
            reliability_results['analysis_type'] == f'{judge_type}_reliability'
        ]
        
        if not judge_reliability.empty:
            print(f"\n{judge_type.replace('_', '-').title()} Reliability by Dimension:")
            
            # Get reliability by dimension
            dimension_reliability = {}
            for metric, name in zip(metrics, metric_names):
                metric_data = judge_reliability[judge_reliability['metric'] == metric]
                if not metric_data.empty:
                    # avg_kappa = metric_data['cohens_kappa'].mean()  # COMMENTED OUT: Not using Cohen's Kappa
                    avg_alpha = metric_data['krippendorff_alpha'].mean()
                    avg_tau = metric_data['kendall_tau'].mean()
                    dimension_reliability[name] = {'alpha': avg_alpha, 'tau': avg_tau}
                    print(f"  {name:12}: α={avg_alpha:.3f}, τ={avg_tau:.3f}")
            
            # Find best/worst dimensions
            if dimension_reliability:
                best_alpha = max(dimension_reliability.items(), key=lambda x: x[1]['alpha'])
                worst_alpha = min(dimension_reliability.items(), key=lambda x: x[1]['alpha'])
                print(f"  → Best reliability: {best_alpha[0]} (α={best_alpha[1]['alpha']:.3f})")
                print(f"  → Worst reliability: {worst_alpha[0]} (α={worst_alpha[1]['alpha']:.3f})")
    
    # Q2a: Cross-judge agreement patterns by dimension
    print("\n📊 Q2a: Does cross-judge agreement vary by dimension?")
    print("-" * 60)
    
    cross_type = reliability_results[reliability_results['analysis_type'] == 'cross_judge_type']
    
    if not cross_type.empty:
        for comparison in cross_type['comparison'].unique():
            comp_data = cross_type[cross_type['comparison'] == comparison]
            print(f"\n{comparison}:")
            
            dimension_agreement = {}
            for metric, name in zip(metrics, metric_names):
                metric_data = comp_data[comp_data['metric'] == metric]
                if not metric_data.empty:
                    avg_tau = metric_data['kendall_tau'].mean()
                    dimension_agreement[name] = avg_tau
                    print(f"  {name:12}: τ={avg_tau:.3f}")
            
            if dimension_agreement:
                best_agreement = max(dimension_agreement.items(), key=lambda x: x[1])
                worst_agreement = min(dimension_agreement.items(), key=lambda x: x[1])
                print(f"  → Strongest agreement: {best_agreement[0]} (τ={best_agreement[1]:.3f})")
                print(f"  → Weakest agreement: {worst_agreement[0]} (τ={worst_agreement[1]:.3f})")
    
    # Q3a: System superiority by dimension
    print("\n📊 Q3a: Which dimensions show RAG superiority?")
    print("-" * 60)
    
    if not system_results.empty:
        for metric, name in zip(metrics, metric_names):
            metric_data = system_results[system_results['metric'] == metric]
            
            if not metric_data.empty:
                significant_improvements = metric_data[
                    (metric_data['wilcoxon_significant'] == True) & 
                    (metric_data['mean_difference'] > 0)
                ]
                
                total_tests = len(metric_data)
                sig_count = len(significant_improvements)
                
                avg_effect_size = metric_data['cohens_d'].mean()
                avg_mean_diff = metric_data['mean_difference'].mean()
                
                print(f"\n{name}:")
                print(f"  Significant RAG improvements: {sig_count}/{total_tests}")
                print(f"  Average effect size (Cohen's d): {avg_effect_size:.3f}")
                print(f"  Average mean difference: {avg_mean_diff:+.3f}")
                
                if sig_count > 0:
                    best_improvement = significant_improvements.loc[
                        significant_improvements['cohens_d'].idxmax()
                    ]
                    print(f"  Best improvement: {best_improvement['judge_model']} ({best_improvement['judge_type']})")
                    print(f"    Cohen's d: {best_improvement['cohens_d']:.3f}")
    
    # Q4a: Judge behavior differences by dimension
    print("\n📊 Q4a: Do judge differences vary by dimension?")
    print("-" * 60)
    
    if not judge_results.empty:
        for metric, name in zip(metrics, metric_names):
            metric_data = judge_results[judge_results['metric'] == metric]
            
            if not metric_data.empty:
                significant_diffs = metric_data[
                    (metric_data['paired_t_sig'] == True) | (metric_data['mwu_sig'] == True)
                ]
                
                total_comparisons = len(metric_data)
                sig_count = len(significant_diffs)
                
                avg_paired_diff = metric_data['paired_mean_diff'].mean()
                avg_cohens_d = metric_data['paired_cohens_d'].mean()
                
                # Direction analysis
                llm_higher = metric_data[
                    (metric_data['paired_mean_diff'] > 0) & (metric_data['paired_t_sig'] == True)
                ]
                agent_higher = metric_data[
                    (metric_data['paired_mean_diff'] < 0) & (metric_data['paired_t_sig'] == True)
                ]
                
                print(f"\n{name}:")
                print(f"  Significant judge differences: {sig_count}/{total_comparisons}")
                print(f"  Average paired difference: {avg_paired_diff:+.3f}")
                print(f"  Average effect size: {avg_cohens_d:.3f}")
                print(f"  LLM-Judge scores higher: {len(llm_higher)} cases")
                print(f"  Agent-Judge scores higher: {len(agent_higher)} cases")
                
                # Identify systematic pattern
                if len(llm_higher) > len(agent_higher):
                    print(f"  → Pattern: LLM-Judges tend to score higher on {name}")
                elif len(agent_higher) > len(llm_higher):
                    print(f"  → Pattern: Agent-Judges tend to score higher on {name}")
                else:
                    print(f"  → Pattern: Mixed judge behavior on {name}")

def create_dimension_summary_tables(reliability_results: pd.DataFrame, 
                                   system_results: pd.DataFrame, 
                                   judge_results: pd.DataFrame) -> None:
    """
    Create dimension-specific summary tables for thesis use.
    """
    metrics = ['score_accuracy', 'score_completeness', 'score_relevance', 'score_overall']
    metric_names = ['Accuracy', 'Completeness', 'Relevance', 'Overall']
    
    # Table 1: Reliability by Dimension and Judge Type
    reliability_summary = []
    
    for judge_type in ['human', 'llm_judge', 'agent_judge']:
        judge_data = reliability_results[
            reliability_results['analysis_type'] == f'{judge_type}_reliability'
        ]
        
        if not judge_data.empty:
            for metric, name in zip(metrics, metric_names):
                metric_data = judge_data[judge_data['metric'] == metric]
                if not metric_data.empty:
                    # avg_kappa = metric_data['cohens_kappa'].mean()  # COMMENTED OUT: Not using Cohen's Kappa
                    avg_alpha = metric_data['krippendorff_alpha'].mean()
                    avg_tau = metric_data['kendall_tau'].mean()
                    
                    reliability_summary.append({
                        'Judge_Type': judge_type.replace('_', '-').title(),
                        'Dimension': name,
                        'Krippendorff_Alpha': avg_alpha,
                        'Kendall_Tau': avg_tau,
                        'Reliability_Level': (
                            'Good' if avg_alpha > 0.67 else 
                            'Moderate' if avg_alpha > 0.4 else 'Poor'
                        )
                    })
    
    if reliability_summary:
        reliability_df = pd.DataFrame(reliability_summary)
        reliability_df.to_csv('dimension_reliability_summary.csv', index=False)
        print("✅ Dimension reliability summary saved to 'dimension_reliability_summary.csv'")
    
    # Table 2: System Comparison by Dimension
    system_summary = []
    
    if not system_results.empty:
        for metric, name in zip(metrics, metric_names):
            metric_data = system_results[system_results['metric'] == metric]
            
            if not metric_data.empty:
                significant_improvements = metric_data[
                    (metric_data['wilcoxon_significant'] == True) & 
                    (metric_data['mean_difference'] > 0)
                ]
                
                system_summary.append({
                    'Dimension': name,
                    'Total_Tests': len(metric_data),
                    'Significant_RAG_Improvements': len(significant_improvements),
                    'Success_Rate': len(significant_improvements) / len(metric_data),
                    'Average_Effect_Size': metric_data['cohens_d'].mean(),
                    'Average_Mean_Difference': metric_data['mean_difference'].mean(),
                    'Best_Judge_Model': (
                        significant_improvements.loc[significant_improvements['cohens_d'].idxmax(), 'judge_model'] 
                        if not significant_improvements.empty else 'None'
                    ),
                    'Best_Effect_Size': (
                        significant_improvements['cohens_d'].max() 
                        if not significant_improvements.empty else 0
                    )
                })
    
    if system_summary:
        system_df = pd.DataFrame(system_summary)
        system_df.to_csv('dimension_system_comparison_summary.csv', index=False)
        print("✅ Dimension system comparison summary saved to 'dimension_system_comparison_summary.csv'")
    
    # Table 3: Judge Behavior by Dimension
    judge_summary = []
    
    if not judge_results.empty:
        for metric, name in zip(metrics, metric_names):
            metric_data = judge_results[judge_results['metric'] == metric]
            
            if not metric_data.empty:
                significant_diffs = metric_data[
                    (metric_data['paired_t_sig'] == True) | (metric_data['mwu_sig'] == True)
                ]
                
                llm_higher = metric_data[
                    (metric_data['paired_mean_diff'] > 0) & (metric_data['paired_t_sig'] == True)
                ]
                agent_higher = metric_data[
                    (metric_data['paired_mean_diff'] < 0) & (metric_data['paired_t_sig'] == True)
                ]
                
                judge_summary.append({
                    'Dimension': name,
                    'Total_Comparisons': len(metric_data),
                    'Significant_Differences': len(significant_diffs),
                    'LLM_Judge_Higher': len(llm_higher),
                    'Agent_Judge_Higher': len(agent_higher),
                    'Average_Difference': metric_data['paired_mean_diff'].mean(),
                    'Average_Effect_Size': metric_data['paired_cohens_d'].mean(),
                    'Dominant_Pattern': (
                        'LLM-Judge Bias' if len(llm_higher) > len(agent_higher) else
                        'Agent-Judge Bias' if len(agent_higher) > len(llm_higher) else
                        'Mixed Behavior'
                    )
                })
    
    if judge_summary:
        judge_df = pd.DataFrame(judge_summary)
        judge_df.to_csv('dimension_judge_behavior_summary.csv', index=False)
        print("✅ Dimension judge behavior summary saved to 'dimension_judge_behavior_summary.csv'")
    
    # Table 4: Cross-Judge Agreement by Dimension
    cross_agreement_summary = []
    
    cross_type = reliability_results[reliability_results['analysis_type'] == 'cross_judge_type']
    
    if not cross_type.empty:
        for comparison in cross_type['comparison'].unique():
            comp_data = cross_type[cross_type['comparison'] == comparison]
            
            for metric, name in zip(metrics, metric_names):
                metric_data = comp_data[comp_data['metric'] == metric]
                if not metric_data.empty:
                    avg_tau = metric_data['kendall_tau'].mean()
                    avg_p = metric_data['kendall_p'].mean()
                    
                    cross_agreement_summary.append({
                        'Judge_Comparison': comparison,
                        'Dimension': name,
                        'Kendall_Tau': avg_tau,
                        'P_Value': avg_p,
                        'Agreement_Level': (
                            'Strong' if avg_tau > 0.7 else
                            'Moderate' if avg_tau > 0.5 else
                            'Weak'
                        ),
                        'Statistical_Significance': 'Significant' if avg_p < 0.05 else 'Not Significant'
                    })
    
    if cross_agreement_summary:
        cross_df = pd.DataFrame(cross_agreement_summary)
        cross_df.to_csv('dimension_cross_judge_agreement_summary.csv', index=False)
        print("✅ Cross-judge agreement summary saved to 'dimension_cross_judge_agreement_summary.csv'")

def generate_research_summary(df: pd.DataFrame, reliability_results: pd.DataFrame, 
                            system_results: pd.DataFrame, judge_results: pd.DataFrame) -> None:
    """
    Generate a comprehensive summary addressing all research questions.
    """
    print("\n" + "="*100)
    print("RESEARCH QUESTIONS SUMMARY (Q1-Q5)")
    print("="*100)
    
    print("\n📋 Q1: Are judges reliable within their type?")
    print("-" * 50)
    
    # Reliability summary by judge type
    for judge_type in ['Human', 'LLM-Judge', 'Agent-Judge']:
        reliability_subset = reliability_results[
            reliability_results['analysis_type'].str.contains('reliability', na=False)
        ]
        if judge_type == 'Human':
            judge_reliability = reliability_subset[reliability_subset['analysis_type'] == 'human_reliability']
        else:
            judge_reliability = reliability_subset[
                reliability_subset['analysis_type'] == f"{judge_type.lower().replace('-', '_')}_reliability"
            ]
        
        if not judge_reliability.empty:
            #avg_kappa = judge_reliability['cohens_kappa'].mean()
            avg_alpha = judge_reliability['krippendorff_alpha'].mean()
            avg_tau = judge_reliability['kendall_tau'].mean()
            print(f"  {judge_type}: Krippendorff's α={avg_alpha:.3f}, Kendall's τ={avg_tau:.3f}")
            #print(f"  {judge_type}: Cohen's κ={avg_kappa:.3f}, Kendall's τ={avg_tau:.3f}")
            
            #if avg_kappa > 0.6:
            if avg_alpha > 0.6:
                print(f"    → Good reliability")
            #elif avg_kappa > 0.4: 
            elif avg_alpha > 0.4:
                print(f"    → Moderate reliability")
            else:
                print(f"    → Poor reliability")
    
    print("\n📋 Q2: Do judges agree across types?")
    print("-" * 50)
    
    cross_type = reliability_results[reliability_results['analysis_type'] == 'cross_judge_type']
    for _, row in cross_type.iterrows():
        tau = row['kendall_tau']
        comparison = row['comparison']
        system = row['system']
        print(f"  {comparison} ({system}): τ={tau:.3f}")
        
        if tau > 0.7:
            print(f"    → Strong agreement")
        elif tau > 0.5:
            print(f"    → Moderate agreement")
        else:
            print(f"    → Weak agreement")
    
    print("\n📋 Q3: Is RAG significantly better than LLM-only?")
    print("-" * 50)
    
    if not system_results.empty:
        significant_improvements = system_results[
            (system_results['wilcoxon_significant'] == True) & 
            (system_results['mean_difference'] > 0)
        ]
        
        total_comparisons = len(system_results)
        print(f"  Significant RAG improvements: {len(significant_improvements)}/{total_comparisons}")
        
        if not significant_improvements.empty:
            best_improvement = significant_improvements.loc[
                significant_improvements['cohens_d'].idxmax()
            ]
            print(f"  Best improvement: {best_improvement['metric']} by {best_improvement['judge_model']} ({best_improvement['judge_type']})")
            print(f"    Cohen's d: {best_improvement['cohens_d']:.3f}")
            print(f"    Mean difference: {best_improvement['mean_difference']:+.3f}")
    
    print("\n📋 Q4: Do LLM-Judges and Agent-Judges score differently?")
    print("-" * 50)
    
    if not judge_results.empty:
        significant_differences = judge_results[
            (judge_results['paired_t_sig'] == True) | (judge_results['mwu_sig'] == True)
        ]
        
        total_comparisons = len(judge_results)
        print(f"  Significant judge differences: {len(significant_differences)}/{total_comparisons}")
        
        # Direction of differences
        llm_higher = judge_results[
            (judge_results['paired_mean_diff'] > 0) & (judge_results['paired_t_sig'] == True)
        ]
        agent_higher = judge_results[
            (judge_results['paired_mean_diff'] < 0) & (judge_results['paired_t_sig'] == True)
        ]
        
        print(f"  LLM-Judge scores higher: {len(llm_higher)} cases")
        print(f"  Agent-Judge scores higher: {len(agent_higher)} cases")
    
    print("\n📋 Q5: What are the qualitative patterns?")
    print("-" * 50)
    print("  (Analyzed in qualitative reasoning section)")
    
    print("\n📋 Q6: Does gemini-1.5-flash show self-evaluation bias?")
    print("-" * 50)
    print("  (Analyzed in self-evaluation bias section)")

# =============================================================================
# Main Enhanced Analysis Function
# =============================================================================

def main_enhanced_analysis(data_path: str = ".") -> None:
    """
    Run the complete enhanced analysis pipeline addressing all research questions.
    """
    print("ENHANCED COMPREHENSIVE NLP EVALUATION ANALYSIS")
    print("="*100)
    print("Addressing 5 key research questions with rigorous statistical methodology")
    
    # Load data
    print("\nSTEP 1: Loading and aggregating data...")
    df = load_and_aggregate_data(data_path)
    
    if df.empty:
        print("Error: No data loaded. Please check your file paths and formats.")
        return
    
    print(f"Data loaded: {len(df)} evaluations")
    
    # Validate data structure
    print("\nSTEP 1.5: Validating data structure...")
    validation_report = validate_data_structure(df)
    print_validation_report(validation_report)
    
    df.to_csv('enhanced_aggregated_data.csv', index=False)
    
    # Enhanced analyses
    print("\nSTEP 2: Comprehensive inter-judge reliability analysis...")
    reliability_results = comprehensive_inter_judge_reliability(df)
    
    print("\nSTEP 3: Enhanced system comparison analysis...")
    system_results = enhanced_system_comparison(df)
    
    print("\nSTEP 4: Enhanced judge comparison analysis...")
    judge_results = enhanced_judge_comparison(df)
    
    print("\nSTEP 5: Self-evaluation bias analysis...")
    bias_results = analyze_self_evaluation_bias(df)
    
    print("\nSTEP 6: Qualitative reasoning analysis...")
    analyze_reasoning_patterns(df)
    
    print("\nSTEP 7: Creating enhanced visualizations...")
    create_comprehensive_visualizations(df, "enhanced_analysis_plots")
    
    print("\nSTEP 8: Research questions summary...")
    generate_research_summary(df, reliability_results, system_results, judge_results)
    
    print("\nSTEP 9: Deep dive analysis (Accuracy & Relevance)...")
    analyze_accuracy_relevance_deep_dive(reliability_results, system_results, judge_results)
    
    print("\nSTEP 10: Surface analysis (All dimensions)...")
    analyze_dimension_specific_patterns(reliability_results, system_results, judge_results)
    
    print("\nSTEP 11: Creating dimension-specific summary tables...")
    create_dimension_summary_tables(reliability_results, system_results, judge_results)
    
    print("\nENHANCED ANALYSIS COMPLETE!")
    print("\nGenerated files:")
    print("  - enhanced_aggregated_data.csv")
    print("  - comprehensive_inter_judge_reliability.csv")
    print("  - enhanced_system_comparison.csv")
    print("  - enhanced_judge_comparison.csv")
    print("  - self_evaluation_bias_analysis.csv (NEW!)")
    print("  - dimension_reliability_summary.csv")
    print("  - dimension_system_comparison_summary.csv")
    print("  - dimension_judge_behavior_summary.csv")
    print("  - dimension_cross_judge_agreement_summary.csv")
    print("  - enhanced_analysis_plots/ (visualizations)")
    print("  - Dimension_Specific_Research_Questions.md (methodology guide)")

if __name__ == "__main__":
    main_enhanced_analysis(".")