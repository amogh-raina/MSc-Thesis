#!/usr/bin/env python3
"""
Thesis-Ready Visualization Suite for NLP Evaluation Analysis

Creates publication-quality visualizations for thesis reporting with clear
axis labels, proper titles, and academic formatting.

Author: Enhanced Analysis Team for Thesis
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set academic style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

def create_thesis_visualizations(df: pd.DataFrame, 
                               reliability_df: pd.DataFrame,
                               system_df: pd.DataFrame,
                               judge_df: pd.DataFrame,
                               output_dir: str = "thesis_plots") -> None:
    """Create all thesis-ready visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreating thesis visualizations in '{output_dir}/'...")
    
    # 1. Judge Performance by Dimension - Bar Plot (Priority 1)
    create_judge_performance_barplot(df, output_dir)
    
    # 2. Reliability Comparison Heatmap (Priority 1) 
    create_reliability_heatmap(reliability_df, output_dir)
    
    # 3. System Performance Effect Sizes (Priority 1)
    create_effect_size_plot(system_df, output_dir)
    
    # 4. Enhanced Human Judge Agreement Heatmap
    create_enhanced_human_heatmap(df, output_dir)
    
    # 5. Cross-Judge Agreement Matrix (Priority 2)
    create_cross_judge_agreement(reliability_df, output_dir)
    
    print("✅ All thesis visualizations created successfully!")

def create_judge_performance_barplot(df: pd.DataFrame, output_dir: str) -> None:
    """Create side-by-side bar plot comparing judge performance by dimension and system."""
    print("  Creating judge performance by dimension bar plot...")
    
    # Filter automated judges only
    automated_df = df[df['judge_type'].isin(['LLM-Judge', 'Agent-Judge'])].copy()
    
    if automated_df.empty:
        print("    No automated judge data found!")
        return
    
    # Calculate means by system, judge type, and dimension
    dimensions = ['score_accuracy', 'score_completeness', 'score_relevance', 'score_overall']
    dim_names = ['Accuracy', 'Completeness', 'Relevance', 'Overall']
    
    # Calculate statistics
    stats_data = []
    for system in ['RAG', 'LLM-only']:
        for judge_type in ['LLM-Judge', 'Agent-Judge']:
            subset = automated_df[(automated_df['system'] == system) & 
                                (automated_df['judge_type'] == judge_type)]
            if not subset.empty:
                for dim, dim_name in zip(dimensions, dim_names):
                    mean_score = subset[dim].mean()
                    std_error = subset[dim].std() / np.sqrt(len(subset))
                    stats_data.append({
                        'System': system,
                        'Judge_Type': judge_type,
                        'Dimension': dim_name,
                        'Mean_Score': mean_score,
                        'Std_Error': std_error
                    })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create subplot for each system
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Judge Performance by Dimension and System', fontsize=16, fontweight='bold')
    
    systems = ['RAG', 'LLM-only']
    colors = ['#3498db', '#e74c3c']  # Blue for LLM-Judge, Red for Agent-Judge
    
    for i, system in enumerate(systems):
        ax = axes[i]
        system_data = stats_df[stats_df['System'] == system]
        
        x = np.arange(len(dim_names))
        width = 0.35
        
        llm_data = system_data[system_data['Judge_Type'] == 'LLM-Judge']
        agent_data = system_data[system_data['Judge_Type'] == 'Agent-Judge']
        
        # Create bars
        bars1 = ax.bar(x - width/2, llm_data['Mean_Score'], width, 
                      yerr=llm_data['Std_Error'], label='LLM-Judge', 
                      color=colors[0], alpha=0.8, capsize=5)
        bars2 = ax.bar(x + width/2, agent_data['Mean_Score'], width,
                      yerr=agent_data['Std_Error'], label='Agent-Judge', 
                      color=colors[1], alpha=0.8, capsize=5)
        
        # Formatting
        ax.set_title(f'{system} System', fontsize=14, fontweight='bold')
        ax.set_xlabel('Evaluation Dimension', fontsize=12)
        ax.set_ylabel('Mean Score (1-5 scale)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(dim_names)
        ax.legend()
        ax.set_ylim(1, 5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'judge_performance_by_dimension.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def create_reliability_heatmap(reliability_df: pd.DataFrame, output_dir: str) -> None:
    """Create reliability comparison heatmap with Krippendorff's Alpha values."""
    print("  Creating reliability comparison heatmap...")
    
    # Filter only reliability data (not cross-judge data)
    reliability_only = reliability_df[reliability_df['analysis_type'].str.contains('reliability', na=False)].copy()
    
    # Extract reliability data
    reliability_data = []
    
    for _, row in reliability_only.iterrows():
        judge_type = row['analysis_type'].replace('_reliability', '')
        judge_type = judge_type.replace('human', 'Human').replace('llm_judge', 'LLM-Judge').replace('agent_judge', 'Agent-Judge')
        
        dimension = row['metric'].replace('score_', '').capitalize()
        alpha_val = row['krippendorff_alpha']
        
        # Only add if we have a valid alpha value
        if pd.notna(alpha_val):
            reliability_data.append({
                'Judge_Type': judge_type,
                'Dimension': dimension,
                'Krippendorff_Alpha': alpha_val
            })
    
    if not reliability_data:
        print("    No reliability data found!")
        return
    
    rel_df = pd.DataFrame(reliability_data)
    
    # Handle duplicate entries by taking the mean (though there shouldn't be any)
    rel_df = rel_df.groupby(['Judge_Type', 'Dimension']).mean().reset_index()
    
    # Create pivot table
    pivot_rel = rel_df.pivot(index='Judge_Type', columns='Dimension', values='Krippendorff_Alpha')
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Custom colormap: Red (poor) -> Yellow (moderate) -> Green (good)
    cmap = sns.diverging_palette(10, 150, s=80, l=55, n=20, center="light", as_cmap=True)
    
    heatmap = sns.heatmap(pivot_rel, annot=True, fmt='.3f', cmap=cmap, 
                         center=0.4, vmin=-0.1, vmax=0.7, square=True,
                         cbar_kws={'label': "Krippendorff's Alpha"}, ax=ax)
    
    ax.set_title("Judge Reliability by Type and Dimension\n(Krippendorff's Alpha)", 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Dimension', fontsize=12)
    ax.set_ylabel('Judge Type', fontsize=12)
    
    # Add reliability interpretation legend
    legend_text = "Interpretation:\nα < 0.4: Poor\n0.4 ≤ α < 0.67: Moderate\nα ≥ 0.67: Good"
    ax.text(1.15, 0.5, legend_text, transform=ax.transAxes, 
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
           verticalalignment='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reliability_comparison_heatmap.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def create_effect_size_plot(system_df: pd.DataFrame, output_dir: str) -> None:
    """Create effect size plot for RAG vs LLM-only system comparison."""
    print("  Creating system performance effect sizes plot...")
    
    if system_df.empty:
        print("    No system comparison data found!")
        return
    
    # Prepare data
    dimensions = ['Accuracy', 'Completeness', 'Relevance', 'Overall']
    effect_data = []
    
    for dim in dimensions:
        dim_data = system_df[system_df['metric'] == f'score_{dim.lower()}']
        if not dim_data.empty:
            for _, row in dim_data.iterrows():
                effect_data.append({
                    'Dimension': dim,
                    'Judge_Model': row['judge_model'],
                    'Judge_Type': row['judge_type'],
                    'Cohens_D': row['cohens_d'],
                    'P_Value': row['wilcoxon_p_value'],
                    'Significant': row['wilcoxon_significant']
                })
    
    if not effect_data:
        print("    No effect size data found!")
        return
    
    effect_df = pd.DataFrame(effect_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    for judge_type, marker, color in [('LLM-Judge', 'o', '#3498db'), ('Agent-Judge', '^', '#e74c3c')]:
        subset = effect_df[effect_df['Judge_Type'] == judge_type]
        significant = subset[subset['Significant'] == True]
        non_significant = subset[subset['Significant'] == False]
        
        # Plot non-significant with hollow markers
        if not non_significant.empty:
            ax.scatter(non_significant['Dimension'], non_significant['Cohens_D'], 
                      marker=marker, s=100, facecolors='none', edgecolors=color,
                      label=f'{judge_type} (ns)', alpha=0.7)
        
        # Plot significant with filled markers
        if not significant.empty:
            ax.scatter(significant['Dimension'], significant['Cohens_D'], 
                      marker=marker, s=100, color=color,
                      label=f'{judge_type} (sig)', alpha=0.9)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    
    # Add effect size interpretation lines
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(y=-0.2, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axhline(y=-0.8, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Formatting
    ax.set_title('System Performance: RAG vs LLM-only\n(Cohen\'s d Effect Sizes)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Evaluation Dimension', fontsize=12)
    ax.set_ylabel('Cohen\'s d (RAG - LLM-only)', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add interpretation text
    interp_text = ("Effect Size Interpretation:\n"
                  "d = ±0.2: Small effect\n"
                  "d = ±0.5: Medium effect\n"
                  "d = ±0.8: Large effect\n"
                  "Positive: RAG > LLM-only\n"
                  "Negative: LLM-only > RAG")
    ax.text(1.05, 0.5, interp_text, transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
           verticalalignment='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'system_performance_effect_sizes.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def create_enhanced_human_heatmap(df: pd.DataFrame, output_dir: str) -> None:
    """Create enhanced human judge agreement heatmap."""
    print("  Creating enhanced human judge agreement heatmap...")
    
    human_data = df[df['judge_type'] == 'Human'].copy()
    if human_data.empty:
        print("    No human judge data found!")
        return
    
    dimensions = ['score_accuracy', 'score_completeness', 'score_relevance', 'score_overall']
    dim_names = ['Accuracy', 'Completeness', 'Relevance', 'Overall']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Human Judge Agreement Analysis by Dimension', fontsize=16, fontweight='bold')
    
    for i, (dim, name) in enumerate(zip(dimensions, dim_names)):
        ax = axes[i//2, i%2]
        
        # Create pivot table
        pivot_data = human_data.pivot_table(
            index='doc_id', 
            columns='judge_model', 
            values=dim, 
            aggfunc='mean'
        ).fillna(0)
        
        if not pivot_data.empty and pivot_data.shape[1] > 1:
            # Calculate correlation matrix
            corr_matrix = pivot_data.corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                       center=0, square=True, ax=ax, vmin=-1, vmax=1,
                       cbar_kws={'label': 'Correlation'})
            
            # Calculate average correlation (excluding diagonal)
            mask = np.eye(corr_matrix.shape[0], dtype=bool)
            avg_corr = corr_matrix.values[~mask].mean()
            
            ax.set_title(f'{name}\nAvg. Correlation: {avg_corr:.3f}', 
                        fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Human Judge', fontsize=10)
        ax.set_ylabel('Human Judge', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'human_judge_agreement_enhanced.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def create_cross_judge_agreement(reliability_df: pd.DataFrame, output_dir: str) -> None:
    """Create cross-judge agreement matrix visualization."""
    print("  Creating cross-judge agreement matrix...")
    
    # Extract cross-judge data
    cross_data = reliability_df[reliability_df['analysis_type'] == 'cross_judge_type'].copy()
    
    if cross_data.empty:
        print("    No cross-judge data found!")
        return
    
    # Prepare data for visualization
    agreement_data = []
    
    for _, row in cross_data.iterrows():
        comparison = row['comparison']
        system = row['system']
        metric = row['metric'].replace('score_', '').capitalize()
        tau = row['kendall_tau']
        
        agreement_data.append({
            'Comparison': comparison,
            'System': system,
            'Dimension': metric,
            'Kendall_Tau': tau
        })
    
    if not agreement_data:
        print("    No agreement data found!")
        return
    
    agree_df = pd.DataFrame(agreement_data)
    
    # Create subplots for each comparison type
    comparisons = agree_df['Comparison'].unique()
    
    fig, axes = plt.subplots(1, len(comparisons), figsize=(15, 5))
    if len(comparisons) == 1:
        axes = [axes]
    
    fig.suptitle('Cross-Judge Agreement by Comparison Type', fontsize=16, fontweight='bold')
    
    for i, comparison in enumerate(comparisons):
        ax = axes[i]
        comp_data = agree_df[agree_df['Comparison'] == comparison]
        
        # Create pivot table
        pivot_comp = comp_data.pivot(index='System', columns='Dimension', values='Kendall_Tau')
        
        # Create heatmap
        sns.heatmap(pivot_comp, annot=True, fmt='.3f', cmap='YlOrRd',
                   vmin=0, vmax=1, square=True, ax=ax,
                   cbar_kws={'label': "Kendall's Tau"})
        
        ax.set_title(f'{comparison}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dimension', fontsize=10)
        ax.set_ylabel('System', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_judge_agreement_matrix.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Load data and create visualizations
    print("Loading analysis results...")
    
    try:
        df = pd.read_csv('enhanced_aggregated_data.csv')
        reliability_df = pd.read_csv('comprehensive_inter_judge_reliability.csv')
        system_df = pd.read_csv('enhanced_system_comparison.csv')
        judge_df = pd.read_csv('enhanced_judge_comparison.csv')
        
        create_thesis_visualizations(df, reliability_df, system_df, judge_df)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV files. Please run Analysis_Enhanced.py first.")
        print(f"Missing file: {e.filename}")
