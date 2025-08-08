from typing import Dict, Any, Optional, List, Union
import asyncio
import json
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextRecall,
    LLMContextPrecisionWithoutReference,
    Faithfulness, 
    FactualCorrectness,
    AnswerAccuracy
)


class RAGASEvalPipeline:
    """
    RAGAS RAG Evaluation Pipeline with enhanced error handling.
    
    Uses LangchainLLMWrapper and follows the SingleTurnSample schema.
    Supports both individual sample evaluation and batch evaluation.
    Enhanced with robust error handling and fallback mechanisms.
    
    Timeouts are configured at the model level in ModelManager.
    """
    
    def __init__(self, evaluator_llm, evaluator_embeddings=None):
        """Initialize RAGAS evaluation pipeline with proper LLM and embedding wrappers."""
        # Wrap LLM and embeddings according to RAGAS requirements
        if not isinstance(evaluator_llm, LangchainLLMWrapper):
            self.evaluator_llm = LangchainLLMWrapper(evaluator_llm)
        else:
            self.evaluator_llm = evaluator_llm
            
        if evaluator_embeddings and not isinstance(evaluator_embeddings, LangchainEmbeddingsWrapper):
            self.evaluator_embeddings = LangchainEmbeddingsWrapper(evaluator_embeddings)
        else:
            self.evaluator_embeddings = evaluator_embeddings
        
        # Initialize metrics with error handling
        self.metrics = {}
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize core RAGAS metrics with error handling"""
        
        # Context Precision
        try:
            self.context_precision = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)
            self.metrics['context_precision'] = self.context_precision
            print("✅ RAGAS Context Precision initialized")
        except Exception as e:
            print(f"❌ RAGAS Context Precision failed: {e}")
            self.context_precision = None
        
        # Context Recall
        try:
            self.context_recall = LLMContextRecall(llm=self.evaluator_llm)
            self.metrics['context_recall'] = self.context_recall
            print("✅ RAGAS Context Recall initialized")
        except Exception as e:
            print(f"❌ RAGAS Context Recall failed: {e}")
            self.context_recall = None
        
        # Faithfulness
        try:
            self.faithfulness = Faithfulness(llm=self.evaluator_llm)
            self.metrics['faithfulness'] = self.faithfulness
            print("✅ RAGAS Faithfulness initialized")
        except Exception as e:
            print(f"❌ RAGAS Faithfulness failed: {e}")
            self.faithfulness = None
        
        # Factual Correctness with parameters
        try:
            self.factual_correctness = FactualCorrectness(
                llm=self.evaluator_llm,
                mode="f1",
                beta=1.0,
                atomicity="low",
                coverage="high"
            )
            self.metrics['factual_correctness'] = self.factual_correctness
            print("✅ RAGAS Factual Correctness initialized")
        except Exception as e:
            print(f"❌ RAGAS Factual Correctness failed: {e}")
            self.factual_correctness = None
        
        # Answer Accuracy
        try:
            self.answer_accuracy = AnswerAccuracy(llm=self.evaluator_llm)
            self.metrics['answer_accuracy'] = self.answer_accuracy
            print("✅ RAGAS Answer Accuracy initialized")
        except Exception as e:
            print(f"❌ RAGAS Answer Accuracy failed: {e}")
            self.answer_accuracy = None
        
        # REMOVED: Noise Sensitivity and Response Relevancy
        self.noise_sensitivity = None
        self.response_relevancy = None
        
        print(f"🔧 RAGAS Pipeline initialized with {len([m for m in self.metrics.values() if m is not None])}/{len(self.metrics)} metrics")

    async def _safe_evaluate_metric(self, metric, sample, metric_name: str) -> Dict[str, Any]:
        """
        Safely evaluate a single metric with error handling
        """
        if metric is None:
            return {
                "score": 0.0,
                "description": f"{metric_name} metric not initialized",
                "error": "Metric initialization failed"
            }
        
        # Log basic input details for debugging
        print(f"🔍 RAGAS {metric_name} - Starting evaluation...")
        print(f"🔍 RAGAS {metric_name} - Question: {sample.user_input[:100]}...")
        
        # Check context size and warn if large
        if hasattr(sample, 'retrieved_contexts') and sample.retrieved_contexts:
            context_length = len(sample.retrieved_contexts[0])
            print(f"🔍 RAGAS {metric_name} - Context length: {context_length}")
            
            if context_length > 15000:
                print(f"⚠️ RAGAS {metric_name} - Large context detected ({context_length} chars)")
        
        try:
            # Special handling for Faithfulness metric with retry logic
            if metric_name == "Faithfulness":
                return await self._evaluate_faithfulness_with_retry(metric, sample)
            
            # Standard evaluation for other metrics
            score = await metric.single_turn_ascore(sample)
            
            print(f"✅ RAGAS {metric_name} - Score: {score}")
            
            return {
                "score": float(score) if score is not None else 0.0,
                "description": f"{metric_name}: {score:.3f}" if score is not None else f"{metric_name}: Failed",
                "evaluation_successful": True
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ RAGAS {metric_name} Error: {error_msg}")
            
            # Basic error categorization
            if "parsing" in error_msg.lower() or "json" in error_msg.lower():
                error_category = "Parsing Error"
            elif "validation" in error_msg.lower() or "schema" in error_msg.lower():
                error_category = "Schema Validation Error"
            elif "timeout" in error_msg.lower():
                error_category = "Timeout Error"
            else:
                error_category = "Unknown Error"
            
            return {
                "score": 0.0,
                "description": f"Error during {metric_name} evaluation: {error_category}",
                "error": error_msg,
                "error_type": type(e).__name__,
                "error_category": error_category
            }

    async def _evaluate_faithfulness_with_retry(self, metric, sample) -> Dict[str, Any]:
        """Enhanced Faithfulness evaluation with retry logic"""
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                score = await metric.single_turn_ascore(sample)
                
                print(f"✅ RAGAS Faithfulness - Score: {score} (attempt {retry_count + 1})")
                
                return {
                    "score": float(score) if score is not None else 0.0,
                    "description": f"Faithfulness: {score:.3f}" if score is not None else "Faithfulness: Failed",
                    "evaluation_successful": True,
                    "retry_attempt": retry_count
                }
                
            except Exception as e:
                error_msg = str(e)
                retry_count += 1
                
                # Check if this is a parsing error that might be retryable
                is_parsing_error = any(keyword in error_msg.lower() for keyword in [
                    "parsing", "json", "validation", "schema", "field required", 
                    "nlistatementoutput", "pydantic", "missing"
                ])
                
                if is_parsing_error and retry_count <= max_retries:
                    print(f"🔄 RAGAS Faithfulness - Parsing error on attempt {retry_count}, retrying...")
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue
                else:
                    print(f"❌ RAGAS Faithfulness - Final error after {retry_count} attempts: {error_msg[:100]}...")
                    
                    # Provide specific error analysis
                    if "field required" in error_msg and "reason" in error_msg:
                        error_analysis = "LLM generated incomplete statements missing 'reason' field"
                    elif "field required" in error_msg and "verdict" in error_msg:
                        error_analysis = "LLM generated incomplete statements missing 'verdict' field"
                    elif "$defs" in error_msg or "properties" in error_msg:
                        error_analysis = "LLM returned JSON schema instead of data"
                    else:
                        error_analysis = "General Faithfulness evaluation error"
                    
                    return {
                        "score": 0.0,
                        "description": f"Faithfulness evaluation failed after {retry_count} attempts",
                        "error": error_msg,
                        "error_type": type(e).__name__,
                        "error_category": "Faithfulness Parsing Error",
                        "error_analysis": error_analysis,
                        "retry_attempts": retry_count
                    }
        
        return {
            "score": 0.0,
            "description": "Faithfulness evaluation failed unexpectedly",
            "error": "Retry logic exhausted"
        }

    async def evaluate_context_recall(self, question: str, reference_answer: str, 
                                    generated_answer: str, context: str) -> Dict[str, Any]:
        """Evaluate context recall with enhanced error handling"""
        sample = SingleTurnSample(
            user_input=question,
            reference=reference_answer,
            response=generated_answer,
            retrieved_contexts=[context]
        )
        return await self._safe_evaluate_metric(
            self.context_recall, sample, "Context Recall"
        )

    async def evaluate_context_precision(self, question: str, reference_answer: str, 
                                       generated_answer: str, context: str) -> Dict[str, Any]:
        """Evaluate context precision with enhanced error handling"""
        sample = SingleTurnSample(
            user_input=question,
            response=generated_answer,
            retrieved_contexts=[context]
        )
        return await self._safe_evaluate_metric(
            self.context_precision, sample, "Context Precision"
        )

    async def evaluate_faithfulness(self, question: str, generated_answer: str, context: str) -> Dict[str, Any]:
        """Evaluate faithfulness with enhanced error handling"""
        sample = SingleTurnSample(
            user_input=question,
            response=generated_answer,
            retrieved_contexts=[context]
        )
        return await self._safe_evaluate_metric(
            self.faithfulness, sample, "Faithfulness"
        )

    async def evaluate_factual_correctness(self, question: str, reference_answer: str, 
                                         generated_answer: str, context: str) -> Dict[str, Any]:
        """Evaluate factual correctness with enhanced error handling"""
        sample = SingleTurnSample(
            user_input=question,
            reference=reference_answer,
            response=generated_answer,
            retrieved_contexts=[context]
        )
        return await self._safe_evaluate_metric(
            self.factual_correctness, sample, "Factual Correctness"
        )
    
    async def evaluate_answer_accuracy(self, question: str, reference_answer: str, 
                                       generated_answer: str) -> Dict[str, Any]:
        """Evaluate answer accuracy with enhanced error handling"""
        sample = SingleTurnSample(
            user_input=question,
            reference=reference_answer,
            response=generated_answer
        )
        return await self._safe_evaluate_metric(
            self.answer_accuracy, sample, "Answer Accuracy"
        )

    async def evaluate_all(self, question: str, reference_answer: str, 
                          generated_answer: str, context: str) -> Dict[str, Any]:
        """Evaluate all available metrics concurrently"""
        
        print(f"🚀 Starting RAGAS evaluation for question: {question[:100]}...")
        
        tasks = []
        task_names = []
        
        # Only add tasks for metrics that were successfully initialized
        if self.context_recall:
            tasks.append(self.evaluate_context_recall(question, reference_answer, generated_answer, context))
            task_names.append("context_recall")
        
        if self.context_precision:
            tasks.append(self.evaluate_context_precision(question, reference_answer, generated_answer, context))
            task_names.append("context_precision")
        
        if self.faithfulness:
            tasks.append(self.evaluate_faithfulness(question, generated_answer, context))
            task_names.append("faithfulness")
        
        if self.factual_correctness:
            tasks.append(self.evaluate_factual_correctness(question, reference_answer, generated_answer, context))
            task_names.append("factual_correctness")

        if self.answer_accuracy:
            tasks.append(self.evaluate_answer_accuracy(question, reference_answer, generated_answer))
            task_names.append("answer_accuracy")
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build results dictionary
        evaluation_results = {}
        successful_evaluations = 0
        failed_evaluations = 0
        
        for i, (task_name, result) in enumerate(zip(task_names, results)):
            if isinstance(result, Exception):
                evaluation_results[task_name] = {
                    "score": 0.0,
                    "error": str(result),
                    "error_type": type(result).__name__
                }
                failed_evaluations += 1
            else:
                evaluation_results[task_name] = result
                if result.get("evaluation_successful", False) or result.get("score", 0) > 0:
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1
        
        # Add metrics that weren't initialized
        all_metrics = ["context_recall", "context_precision", 
                      "faithfulness", "factual_correctness", "answer_accuracy"]
        
        for metric_name in all_metrics:
            if metric_name not in evaluation_results:
                evaluation_results[metric_name] = {
                    "score": 0.0,
                    "description": f"{metric_name} metric not available",
                    "error": "Metric not initialized"
                }
                failed_evaluations += 1
        
        # Create summary
        score_summary = []
        for metric_name, result in evaluation_results.items():
            score = result.get("score", 0.0)
            score_summary.append(f"{metric_name}: {score:.3f}")
        
        print(f"✅ RAGAS evaluation completed. Scores: {score_summary}")
        print(f"📊 Summary: {successful_evaluations} successful, {failed_evaluations} failed")
        
        return evaluation_results

    async def batch_evaluate(self, samples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Evaluate multiple samples in batch"""
        if not samples:
            return []
        
        print(f"🚀 Starting RAGAS batch evaluation for {len(samples)} samples...")
        
        results = []
        for i, sample in enumerate(samples):
            try:
                print(f"📝 Processing sample {i+1}/{len(samples)}")
                result = await self.evaluate_all(
                    question=sample.get("question", ""),
                    reference_answer=sample.get("reference_answer", ""),
                    generated_answer=sample.get("generated_answer", ""),
                    context=sample.get("context", "")
                )
                
                results.append({
                    "sample_index": i,
                    "evaluation": result,
                    "status": "completed"
                })
                
            except Exception as e:
                print(f"❌ Error processing sample {i+1}: {str(e)}")
                results.append({
                    "sample_index": i,
                    "evaluation": {},
                    "status": "failed",
                    "error": str(e)
                })
        
        print(f"✅ RAGAS batch evaluation completed: {len(results)} samples processed")
        return results

    def batch_evaluate_dataset(self, samples: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate dataset using asyncio with progress tracking"""
        if not samples:
            return {"results": [], "summary": {}}
        
        # Run async evaluation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(self.batch_evaluate(samples))
        finally:
            loop.close()
        
        # Calculate summary statistics
        successful_count = sum(1 for r in results if r.get("status") == "completed")
        failed_count = len(results) - successful_count
        
        summary = {
            "total_samples": len(samples),
            "successful_evaluations": successful_count,
            "failed_evaluations": failed_count,
            "success_rate": successful_count / len(samples) if samples else 0
        }
        
        return {
            "results": results,
            "summary": summary
        }

    def get_aggregate_scores(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate scores from individual evaluation results"""
        if not evaluation_results:
            return {}
            
        # Initialize score lists for core metrics
        score_lists = {
            "context_recall": [],
            "context_precision": [],
            "faithfulness": [],
            "factual_correctness": [],
            "answer_accuracy": []
        }
        
        for result in evaluation_results:
            if "evaluation" in result:
                eval_data = result["evaluation"]
                
                # Extract scores with proper type checking
                for metric_name, score_list in score_lists.items():
                    if metric_name in eval_data and "score" in eval_data[metric_name]:
                        score = eval_data[metric_name]["score"]
                        # Only include valid numeric scores
                        if isinstance(score, (int, float)) and not isinstance(score, bool) and score > 0:
                            score_list.append(score)
        
        # Calculate averages only for metrics with valid scores
        aggregates = {}
        for metric_name, scores in score_lists.items():
            if scores:
                avg_name = f"avg_{metric_name}"
                aggregates[avg_name] = sum(scores) / len(scores)
                
        return aggregates

    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available RAGAS metrics"""
        return {
            "context_recall": "Measures how much of the relevant information from the reference is captured in the retrieved context. Higher values indicate better retrieval completeness.",
            "context_precision": "Measures the precision of retrieved context by evaluating what proportion of the retrieved context is relevant to the generated answer. Uses LLM to compare retrieved contexts with the response.",
            "faithfulness": "Measures the factual consistency of the generated answer against the retrieved context. Higher values indicate less hallucination.",
            "factual_correctness": "Measures the factual accuracy of the generated answer compared to the reference answer using high atomicity and coverage for detailed legal analysis.",
            "answer_accuracy": "Measures the accuracy of the generated answer compared to the reference answer. Higher values indicate more accurate answers."
        }

    def get_factual_correctness_config(self) -> Dict[str, Any]:
        """Get the current FactualCorrectness configuration for transparency"""
        return {
            "mode": "f1",
            "beta": 1.0,
            "atomicity": "low",
            "coverage": "high",
            "description": "Optimized for legal Q&A with detailed claim decomposition and comprehensive coverage"
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get current status of all RAGAS metrics"""
        return {
            "context_recall": self.context_recall is not None,
            "context_precision": self.context_precision is not None,
            "faithfulness": self.faithfulness is not None,
            "factual_correctness": self.factual_correctness is not None,
            "answer_accuracy": self.answer_accuracy is not None,
            "total_metrics": len([m for m in self.metrics.values() if m is not None]),
            "llm_wrapper": type(self.evaluator_llm).__name__,
            "embeddings_wrapper": type(self.evaluator_embeddings).__name__ if self.evaluator_embeddings else "None"
        }

# Backward compatibility alias
RAGAS_Eval_Pipeline = RAGASEvalPipeline


