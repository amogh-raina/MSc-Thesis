# CLI entry point for LLM Judge (Parametric Knowledge Only)
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

# Import LLM Judge components
from .llm_judge_evaluator import LLMJudgeEvaluator
from .utils import load_file

def main():
    """Main CLI entry point for LLM Judge"""
    parser = argparse.ArgumentParser(description="LLM Judge - Evaluate legal answers using pure parametric knowledge (no web search)")
    
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
        print(f"🧠 Using LLM Judge: {args.provider}/{selected_model}")
        print(f"📄 Input file: {args.file_path}")
        print("🚀 Method: Parametric knowledge only (no web search)")
    
    # Create LLM instance
    try:
        llm = ModelManager.create_llm(args.provider, selected_model)
        if not llm:
            print(f"❌ Error: Failed to create LLM instance for {args.provider}/{selected_model}")
            return 1
    except Exception as e:
        print(f"❌ Error creating LLM: {str(e)}")
        return 1
    
    # Load trios
    try:
        if args.verbose:
            print(f"📁 Loading file...")
        
        all_trios = load_file(args.file_path)
        if not all_trios:
            print("❌ No valid trios found in file")
            return 1
        
        print(f"✅ Loaded {len(all_trios)} trios from {args.file_path}")
        
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return 1
    
    # Run LLM Judge evaluation
    try:
        if args.verbose:
            print(f"🧠 Starting LLM Judge evaluation...")
        
        # Create LLM judge evaluator
        llm_judge = LLMJudgeEvaluator(llm)
        
        evaluations = []
        
        for i, trio in enumerate(all_trios):
            if args.verbose:
                print(f"🔄 Evaluating trio {i+1}/{len(all_trios)}")
            
            try:
                # Run evaluation for this trio
                evaluation_result = llm_judge.evaluate_all(
                    question=trio["question"],
                    reference_answer=trio["reference_answer"],
                    generated_answer=trio["generated_answer"]
                )
                
                # Format result
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
                    "evaluation_method": "LLM Judge",
                    "llm_provider": args.provider,
                    "llm_model": selected_model,
                    "timestamp": datetime.now().isoformat()
                }
                
                evaluations.append(result)
                
            except Exception as e:
                # Handle individual evaluation errors
                print(f"⚠️  Error evaluating trio {i+1}: {str(e)}")
                error_result = {
                    "trio_index": i,
                    "question": trio["question"],
                    "reference_answer": trio["reference_answer"],  # Add reference answer
                    "generated_answer": trio["generated_answer"],  # Add generated answer
                    "error": f"LLM Judge evaluation failed: {str(e)}",
                    "scores": {"error": "Evaluation failed"},
                    "evaluation_method": "LLM Judge",
                    "llm_provider": args.provider,
                    "llm_model": selected_model,
                    "timestamp": datetime.now().isoformat()
                }
                evaluations.append(error_result)
        
        # Generate output path
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            input_name = Path(args.file_path).stem
            output_path = f"llm_judge_results_{input_name}_{timestamp}.json"
        
        # Save results
        successful = [e for e in evaluations if "scores" in e and "error" not in e.get("scores", {})]
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "input_file": args.file_path,
                "evaluation_method": "LLM Judge",
                "llm_provider": args.provider,
                "llm_model": selected_model,
                "total_evaluations": len(evaluations),
                "successful_evaluations": len(successful),
                "error_evaluations": len(evaluations) - len(successful)
            },
            "evaluations": evaluations,
            "summary": generate_summary(evaluations)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Display summary
        print(f"✅ LLM Judge evaluation completed!")
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
    successful = [e for e in evaluations if "scores" in e and "error" not in e.get("scores", {})]
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
    
    return {
        "successful": len(successful),
        "errors": errors,
        "average_scores": averages,
        "evaluation_method": "LLM Judge - Parametric Knowledge Only"
    }

def display_summary(summary):
    """Display summary in a nice format"""
    print(f"\n📊 LLM Judge Summary:")
    print(f"   Successful evaluations: {summary.get('successful', 0)}")
    print(f"   Errors: {summary.get('errors', 0)}")
    print(f"   Method: {summary.get('evaluation_method', 'LLM Judge')}")
    
    if summary.get('average_scores'):
        print(f"   Average scores:")
        for metric, score in summary['average_scores'].items():
            print(f"     {metric}: {score:.3f}")

if __name__ == "__main__":
    exit(main()) 