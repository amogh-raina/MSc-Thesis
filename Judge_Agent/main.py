# CLI entry point for Judge Agent 
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add the parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import from Main folder (existing architecture)
from Main.core.model_manager import ModelManager
from Main.config.settings import *

# Import Judge Agent components
from .workflow import run_judge_evaluation

def main():
    """Main CLI entry point for Judge Agent"""
    parser = argparse.ArgumentParser(description="Judge Agent - Evaluate legal answers using LLM and optional web search")
    
    parser.add_argument("file_path", help="Path to CSV/Excel file with question/reference_answer/generated_answer columns")
    parser.add_argument("--provider", default="OpenAI", help="LLM provider (OpenAI, NVIDIA, Mistral, Google, etc.)")
    parser.add_argument("--model", help="LLM model name (if not specified, uses first available model for provider)")
    parser.add_argument("--output", help="Output JSON file path (default: auto-generated)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Validate file path
    if not os.path.exists(args.file_path):
        print(f"❌ Error: File '{args.file_path}' not found")
        return 1
    
    # Get available providers
    providers = ModelManager.get_available_llm_providers()
    
    if not providers:
        print("❌ Error: No LLM providers available. Please check your API keys.")
        print("Required environment variables: OPENAI_API_KEY, NVIDIA_API_KEY, MISTRAL_API_KEY, etc.")
        return 1
    
    # Validate provider
    if args.provider not in providers:
        print(f"❌ Error: Provider '{args.provider}' not available.")
        print(f"Available providers: {list(providers.keys())}")
        return 1
    
    # Select model
    available_models = providers[args.provider]["models"]
    if args.model:
        if args.model not in available_models:
            print(f"❌ Error: Model '{args.model}' not available for provider '{args.provider}'")
            print(f"Available models: {available_models}")
            return 1
        selected_model = args.model
    else:
        selected_model = available_models[0]  # Use first available model
    
    if args.verbose:
        print(f"🤖 Using LLM: {args.provider}/{selected_model}")
        print(f"📄 Input file: {args.file_path}")
        tavily_status = "✅ Available" if os.getenv("TAVILY_API_KEY") else "❌ Not available (web search disabled)"
        print(f"🔍 Web search: {tavily_status}")
    
    # Create LLM instance
    try:
        llm = ModelManager.create_llm(args.provider, selected_model)
        if not llm:
            print(f"❌ Error: Failed to create LLM instance for {args.provider}/{selected_model}")
            return 1
    except Exception as e:
        print(f"❌ Error creating LLM: {str(e)}")
        return 1
    
    # Run evaluation
    try:
        if args.verbose:
            print(f"🚀 Starting evaluation...")
        
        final_state = run_judge_evaluation(args.file_path, llm)
        
        # Extract results
        evaluations = final_state.get("all_evaluations", [])
        
        if not evaluations:
            print("❌ No evaluations completed")
            return 1
        
        # Generate output path
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_name = Path(args.file_path).stem
            output_path = f"judge_results_{input_name}_{timestamp}.json"
        
        # Save results
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "input_file": args.file_path,
                "llm_provider": args.provider,
                "llm_model": selected_model,
                "total_evaluations": len(evaluations),
                "web_search_available": bool(os.getenv("TAVILY_API_KEY"))
            },
            "evaluations": evaluations,
            "summary": generate_summary(evaluations)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Display summary
        print(f"✅ Evaluation completed!")
        print(f"📊 Processed {len(evaluations)} question-answer pairs")
        print(f"💾 Results saved to: {output_path}")
        
        if args.verbose:
            display_summary(results["summary"])
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

def generate_summary(evaluations):
    """Generate summary statistics from evaluations"""
    if not evaluations:
        return {}
    
    # Count successful evaluations (those with scores)
    successful = [e for e in evaluations if "scores" in e and "error" not in e["scores"]]
    errors = len(evaluations) - len(successful)
    
    if not successful:
        return {"errors": errors, "successful": 0}
    
    # Calculate average scores
    scores = {}
    for evaluation in successful:
        for metric, value in evaluation["scores"].items():
            if isinstance(value, (int, float)):
                if metric not in scores:
                    scores[metric] = []
                scores[metric].append(value)
    
    averages = {metric: sum(values)/len(values) for metric, values in scores.items()}
    
    # Count web searches
    web_searches = sum(1 for e in successful if e.get("web_search_used"))
    
    return {
        "successful": len(successful),
        "errors": errors,
        "average_scores": averages,
        "web_searches_used": web_searches,
        "web_search_rate": web_searches / len(successful) if successful else 0
    }

def display_summary(summary):
    """Display summary in a nice format"""
    print(f"\n📊 Summary:")
    print(f"   Successful evaluations: {summary.get('successful', 0)}")
    print(f"   Errors: {summary.get('errors', 0)}")
    
    if summary.get('average_scores'):
        print(f"   Average scores:")
        for metric, score in summary['average_scores'].items():
            print(f"     {metric}: {score:.3f}")
    
    if summary.get('web_searches_used'):
        print(f"   Web searches used: {summary['web_searches_used']} ({summary.get('web_search_rate', 0)*100:.1f}%)")

if __name__ == "__main__":
    exit(main()) 