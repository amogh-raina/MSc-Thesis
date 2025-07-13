from typing import Dict, Any, Optional, List, Union
import asyncio
import json
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextRecall,
    LLMContextPrecisionWithoutReference,
    # ContextEntityRecall,
    Faithfulness, 
    FactualCorrectness,
    AnswerAccuracy
)

### RAGAS RAG EVALUATION FRAMEWORK - ENHANCED VERSION
# 
# This framework uses RAGAS metrics following the official documentation patterns.
# Based on: https://docs.ragas.io/en/stable/
# 
# Key improvements:
# - Enhanced error handling for parsing failures
# - Fallback mechanisms for problematic metrics
# - Better model compatibility
# - Robust JSON parsing with multiple strategies
# 
# Key metrics:
# - Context Recall: Measures completeness of retrieved context vs reference
# - Context Precision: Measures precision of retrieved context vs generated response
# - Context Entity Recall: Measures entity-based recall between context and reference
# - Faithfulness: Measures factual consistency of answer vs context  
# - Factual Correctness: Measures factual accuracy of answer vs reference
# - Answer Accuracy: Measures accuracy of generated answer vs reference
# - Response Relevancy: Measures relevance of answer to question
# - Noise Sensitivity: Measures robustness to irrelevant context

class RAGASEvalPipeline:
    """
    Enhanced RAGAS RAG Evaluation Pipeline with improved error handling.
    
    Uses proper LangchainLLMWrapper and follows the SingleTurnSample schema.
    Supports both individual sample evaluation and batch evaluation using EvaluationDataset.
    Enhanced with robust error handling and fallback mechanisms.
    """
    
    def __init__(self, evaluator_llm, evaluator_embeddings=None):
        """
        Initialize RAGAS evaluation pipeline with proper LLM and embedding wrappers.
        
        Args:
            evaluator_llm: LangChain LLM instance (will be wrapped in LangchainLLMWrapper)
            evaluator_embeddings: LangChain embeddings instance (will be wrapped in LangchainEmbeddingsWrapper)
        """
        # Wrap LLM and embeddings according to RAGAS requirements
        if not isinstance(evaluator_llm, LangchainLLMWrapper):
            self.evaluator_llm = LangchainLLMWrapper(evaluator_llm)
        else:
            self.evaluator_llm = evaluator_llm
            
        if evaluator_embeddings and not isinstance(evaluator_embeddings, LangchainEmbeddingsWrapper):
            self.evaluator_embeddings = LangchainEmbeddingsWrapper(evaluator_embeddings)
        else:
            self.evaluator_embeddings = evaluator_embeddings
        
        print(f"🔧 RAGAS Initialization - LLM type: {type(evaluator_llm)}")
        if evaluator_embeddings:
            print(f"🔧 RAGAS Initialization - Embeddings type: {type(evaluator_embeddings)}")
        
        # Initialize metrics with proper LLM instances and error handling
        self.metrics = {}
        self._initialize_metrics()
        
        print("✅ RAGAS Pipeline initialized successfully")

    def _initialize_metrics(self):
        """Initialize only the 5 core RAGAS metrics with error handling"""
        
        # Context Precision - usually most reliable
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
        
        # # Context Entity Recall
        # try:
        #     self.context_entity_recall = ContextEntityRecall(llm=self.evaluator_llm)
        #     self.metrics['context_entity_recall'] = self.context_entity_recall
        #     print("✅ RAGAS Context Entity Recall initialized")
        # except Exception as e:
        #     print(f"❌ RAGAS Context Entity Recall failed: {e}")
        #     self.context_entity_recall = None
        
        # Faithfulness
        try:
            self.faithfulness = Faithfulness(llm=self.evaluator_llm)
            self.metrics['faithfulness'] = self.faithfulness
            print("✅ RAGAS Faithfulness initialized")
        except Exception as e:
            print(f"❌ RAGAS Faithfulness failed: {e}")
            self.faithfulness = None
        
        # Factual Correctness with optimized parameters
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
        
        # REMOVED: Noise Sensitivity (was causing timeouts and consuming too many LLM calls)
        self.noise_sensitivity = None
        print("⚠️ RAGAS Noise Sensitivity DISABLED - was causing timeouts")
        
        # REMOVED: Response Relevancy (requires embeddings and not in core 5 metrics)
        self.response_relevancy = None
        print("⚠️ RAGAS Response Relevancy DISABLED - not in core 5 metrics")

    async def _safe_evaluate_metric(self, metric, sample, metric_name: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Safely evaluate a single metric with timeout and error handling
        
        Args:
            metric: The RAGAS metric instance
            sample: SingleTurnSample instance
            metric_name: Name of the metric for logging
            timeout: Timeout in seconds
            
        Returns:
            Dict with score and metadata or error info
        """
        if metric is None:
            return {
                "score": 0.0,
                "description": f"{metric_name} metric not initialized",
                "error": "Metric initialization failed",
                "llm_reasoning": "N/A - Metric not initialized"
            }
        
        try:
            print(f"🔍 RAGAS {metric_name} - Starting evaluation...")
            
            # Log input details for debugging
            print(f"🔍 RAGAS {metric_name} - Question: {sample.user_input[:100]}...")
            if hasattr(sample, 'reference') and sample.reference:
                print(f"🔍 RAGAS {metric_name} - Reference: {sample.reference[:100]}...")
            if hasattr(sample, 'response') and sample.response:
                print(f"🔍 RAGAS {metric_name} - Generated: {sample.response[:100]}...")
            if hasattr(sample, 'retrieved_contexts') and sample.retrieved_contexts:
                print(f"🔍 RAGAS {metric_name} - Context length: {len(sample.retrieved_contexts[0])}")
            
            # Capture LLM calls by temporarily wrapping the LLM
            llm_calls = []
            original_llm = metric.llm
            
            # Create a wrapper to capture LLM interactions
            class LLMCallCapture:
                def __init__(self, original_llm):
                    self.original_llm = original_llm
                    self.calls = []
                
                def __getattr__(self, name):
                    return getattr(self.original_llm, name)
                
                async def agenerate(self, *args, **kwargs):
                    # Capture the input
                    if args and hasattr(args[0], '__iter__'):
                        for message_batch in args[0]:
                            if hasattr(message_batch, '__iter__'):
                                for message in message_batch:
                                    if hasattr(message, 'content'):
                                        self.calls.append(f"RAGAS LLM Input: {repr(message)}")
                                        print(f"RAGAS LLM Input: {repr(message)}")
                    
                    # Call the original method
                    result = await self.original_llm.agenerate(*args, **kwargs)
                    
                    # Capture the output
                    if hasattr(result, 'generations'):
                        for gen_batch in result.generations:
                            for gen in gen_batch:
                                if hasattr(gen, 'text'):
                                    self.calls.append(f"RAGAS LLM Output: {gen.text}")
                                    print(f"RAGAS LLM Output: {gen.text}")
                    
                    return result
                
                def generate(self, *args, **kwargs):
                    # Capture the input
                    if args and hasattr(args[0], '__iter__'):
                        for message_batch in args[0]:
                            if hasattr(message_batch, '__iter__'):
                                for message in message_batch:
                                    if hasattr(message, 'content'):
                                        self.calls.append(f"RAGAS LLM Input: {repr(message)}")
                                        print(f"RAGAS LLM Input: {repr(message)}")
                    
                    # Call the original method
                    result = self.original_llm.generate(*args, **kwargs)
                    
                    # Capture the output
                    if hasattr(result, 'generations'):
                        for gen_batch in result.generations:
                            for gen in gen_batch:
                                if hasattr(gen, 'text'):
                                    self.calls.append(f"RAGAS LLM Output: {gen.text}")
                                    print(f"RAGAS LLM Output: {gen.text}")
                    
                    return result
            
            # Wrap the LLM
            capture_wrapper = LLMCallCapture(original_llm)
            metric.llm = capture_wrapper
            
            # Use asyncio.wait_for to implement timeout
            score = await asyncio.wait_for(
                metric.single_turn_ascore(sample),
                timeout=timeout
            )
            
            # Restore original LLM
            metric.llm = original_llm
            
            print(f"✅ RAGAS {metric_name} - Score: {score}")
            
            # Compile LLM reasoning
            llm_reasoning = "\n".join(capture_wrapper.calls) if capture_wrapper.calls else "No LLM calls captured"
            
            return {
                "score": float(score) if score is not None else 0.0,
                "description": f"{metric_name}: {score:.3f}" if score is not None else f"{metric_name}: Failed",
                "raw_response": llm_reasoning,
                "llm_reasoning": llm_reasoning,
                "evaluation_successful": True,
                "input_question": sample.user_input[:200] + "..." if len(sample.user_input) > 200 else sample.user_input,
                "input_context_length": len(sample.retrieved_contexts[0]) if hasattr(sample, 'retrieved_contexts') and sample.retrieved_contexts else 0,
                "input_reference_length": len(sample.reference) if hasattr(sample, 'reference') and sample.reference else 0,
                "input_response_length": len(sample.response) if hasattr(sample, 'response') and sample.response else 0
            }
            
        except asyncio.TimeoutError:
            print(f"⏰ RAGAS {metric_name} - Timeout after {timeout}s")
            return {
                "score": 0.0,
                "description": f"{metric_name} evaluation timed out",
                "error": f"Evaluation timeout after {timeout}s",
                "error_type": "TimeoutError",
                "llm_reasoning": "Evaluation timed out - no reasoning captured"
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ RAGAS {metric_name} Error: {error_msg}")
            
            # Enhanced error categorization
            if "parsing" in error_msg.lower() or "json" in error_msg.lower():
                error_category = "Parsing Error"
                suggestion = "LLM output format incompatible with RAGAS schema"
            elif "validation" in error_msg.lower() or "schema" in error_msg.lower():
                error_category = "Schema Validation Error"
                suggestion = "LLM response doesn't match expected format"
            elif "timeout" in error_msg.lower():
                error_category = "Timeout Error"
                suggestion = "Evaluation took too long"
            else:
                error_category = "Unknown Error"
                suggestion = "Check LLM compatibility and input quality"
            
            return {
                "score": 0.0,
                "description": f"Error during {metric_name} evaluation: {error_category}",
                "error": error_msg,
                "error_type": type(e).__name__,
                "error_category": error_category,
                "suggestion": suggestion,
                "detailed_error": str(e),
                "llm_reasoning": f"Error during evaluation: {error_msg}"
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

    # async def evaluate_context_entity_recall(self, question: str, reference_answer: str, 
    #                                        generated_answer: str, context: str) -> Dict[str, Any]:
    #     """Evaluate context entity recall with enhanced error handling"""
    #     sample = SingleTurnSample(
    #         user_input=question,
    #         reference=reference_answer,
    #         response=generated_answer,
    #         retrieved_contexts=[context]
    #     )
    #     return await self._safe_evaluate_metric(
    #         self.context_entity_recall, sample, "Context Entity Recall"
    #     )

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

    # async def evaluate_noise_sensitivity(self, question: str, reference_answer: str, 
    #                                    generated_answer: str, context: str) -> Dict[str, Any]:
    #     """Evaluate noise sensitivity with enhanced error handling and shorter timeout"""
    #     sample = SingleTurnSample(
    #         user_input=question,
    #         reference=reference_answer,
    #         response=generated_answer,
    #         retrieved_contexts=[context]
    #     )
    #     # Use shorter timeout for noise sensitivity as it's been problematic
    #     return await self._safe_evaluate_metric(
    #         self.noise_sensitivity, sample, "Noise Sensitivity", timeout=30
    #     )

    # async def evaluate_response_relevancy(self, question: str, generated_answer: str, 
    #                                     context: str) -> Dict[str, Any]:
    #     """Evaluate response relevancy with enhanced error handling"""
    #     if not self.response_relevancy:
    #         return {
    #             "score": 0.0,
    #             "description": "Response Relevancy evaluation requires embeddings model",
    #             "error": "No embeddings model provided"
    #         }
            
    #     sample = SingleTurnSample(
    #         user_input=question,
    #         response=generated_answer,
    #         retrieved_contexts=[context]
    #     )
    #     return await self._safe_evaluate_metric(
    #         self.response_relevancy, sample, "Response Relevancy"
    #     )

    async def evaluate_all(self, question: str, reference_answer: str, 
                          generated_answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluate all available metrics for a single sample with enhanced error handling.
        """
        print(f"🚀 Starting RAGAS evaluation for question: {question[:100]}...")
        
        # Define all evaluation tasks
        tasks = []
        task_names = []
        
        # Only add tasks for metrics that were successfully initialized
        if self.context_recall:
            tasks.append(self.evaluate_context_recall(question, reference_answer, generated_answer, context))
            task_names.append("context_recall")
        
        if self.context_precision:
            tasks.append(self.evaluate_context_precision(question, reference_answer, generated_answer, context))
            task_names.append("context_precision")
        
        # if self.context_entity_recall:
        #     tasks.append(self.evaluate_context_entity_recall(question, reference_answer, generated_answer, context))
        #     task_names.append("context_entity_recall")
        
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
        
        # Add metrics that weren't initialized (only the 5 core metrics) ["context_entity_recall"]
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
        """
        Evaluate multiple samples individually with enhanced error handling.
        """
        results = []
        print(f"🚀 Starting RAGAS batch evaluation for {len(samples)} samples...")
        
        for i, sample in enumerate(samples):
            try:
                print(f"📝 Processing sample {i+1}/{len(samples)}")
                evaluation = await self.evaluate_all(
                    question=sample["question"],
                    reference_answer=sample["reference_answer"],
                    generated_answer=sample["generated_answer"],
                    context=sample.get("context", "")
                )
                result = {
                    "sample_id": i,
                    "question": sample["question"],
                    "reference_answer": sample["reference_answer"],
                    "generated_answer": sample["generated_answer"],
                    "context": sample.get("context", ""),
                    "evaluation": evaluation
                }
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing sample {i+1}: {str(e)}")
                error_result = {
                    "sample_id": i,
                    "question": sample.get("question", ""),
                    "error": str(e),
                    "evaluation": {
                        "context_recall": {"score": 0.0, "error": str(e)},
                        "context_precision": {"score": 0.0, "error": str(e)},
                        # "context_entity_recall": {"score": 0.0, "error": str(e)},
                        "faithfulness": {"score": 0.0, "error": str(e)},
                        "factual_correctness": {"score": 0.0, "error": str(e)},
                        "answer_accuracy": {"score": 0.0, "error": str(e)}
                    }
                }
                results.append(error_result)
        
        print(f"✅ RAGAS batch evaluation completed: {len(results)} samples processed")
        return results

    def batch_evaluate_dataset(self, samples: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate using RAGAS's built-in batch evaluation with EvaluationDataset.
        Enhanced with better error handling.
        """
        try:
            print(f"🚀 Starting RAGAS dataset evaluation for {len(samples)} samples...")
            
            # Convert samples to RAGAS dataset format
            dataset_items = []
            for sample in samples:
                dataset_items.append({
                    "user_input": sample["question"],
                    "reference": sample["reference_answer"], 
                    "response": sample["generated_answer"],
                    "retrieved_contexts": [sample.get("context", "")]
                })
            
            # Create EvaluationDataset
            dataset = EvaluationDataset.from_list(dataset_items)
            
            # Prepare metrics list (only include successfully initialized metrics)
            metrics = []
            for metric_name, metric in self.metrics.items():
                if metric is not None:
                    metrics.append(metric)
                    print(f"✅ Including {metric_name} in batch evaluation")
                else:
                    print(f"⚠️ Skipping {metric_name} - not initialized")
            
            if not metrics:
                return {
                    "error": "No metrics available for evaluation",
                    "aggregate_scores": {},
                    "individual_results": None,
                    "dataset_size": len(samples)
                }
            
            # Run evaluation
            print(f"🔧 Running RAGAS evaluation with {len(metrics)} metrics...")
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings
            )
            
            print("✅ RAGAS dataset evaluation completed successfully")
            return {
                "aggregate_scores": dict(result),
                "individual_results": result.to_pandas().to_dict('records') if hasattr(result, 'to_pandas') else None,
                "dataset_size": len(samples),
                "metrics_used": len(metrics)
            }
            
        except Exception as e:
            print(f"❌ RAGAS dataset evaluation failed: {str(e)}")
            return {
                "error": f"Batch evaluation failed: {str(e)}",
                "aggregate_scores": {},
                "individual_results": None,
                "dataset_size": len(samples)
            }

    def get_aggregate_scores(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregate scores from individual evaluation results with enhanced handling.
        """
        if not evaluation_results:
            return {}
            
        # Initialize score lists for the 5 core metrics only
        score_lists = {
            "context_recall": [],
            "context_precision": [],
            # "context_entity_recall": [],
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
                print(f"📊 {metric_name}: {len(scores)} valid scores, avg = {aggregates[avg_name]:.3f}")
            else:
                print(f"⚠️ {metric_name}: No valid scores found")
                
        return aggregates

    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available RAGAS metrics."""
        return {
            "context_recall": "Measures how much of the relevant information from the reference is captured in the retrieved context. Higher values indicate better retrieval completeness.",
            "context_precision": "Measures the precision of retrieved context by evaluating what proportion of the retrieved context is relevant to the generated answer. Uses LLM to compare retrieved contexts with the response.",
            # "context_entity_recall": "Measures recall based on entities present in ground truth and context. Useful for entity-focused legal applications.",
            "faithfulness": "Measures the factual consistency of the generated answer against the retrieved context. Higher values indicate less hallucination.",
            "factual_correctness": "Measures the factual accuracy of the generated answer compared to the reference answer using high atomicity and coverage for detailed legal analysis.",
            # "noise_sensitivity": "Measures how robust the system is to irrelevant or noisy information in the retrieved context.",
            # "response_relevancy": "Measures how relevant and focused the generated answer is to the input question. Higher values indicate more relevant responses."
            "answer_accuracy": "Measures the accuracy of the generated answer compared to the reference answer. Higher values indicate more accurate answers."
        }

    def get_factual_correctness_config(self) -> Dict[str, Any]:
        """Get the current FactualCorrectness configuration for transparency."""
        return {
            "mode": "f1",
            "beta": 1.0,
            "atomicity": "low",
            "coverage": "high",
            "description": "Optimized for legal Q&A with detailed claim decomposition and comprehensive coverage"
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get the current system status and metric availability."""
        return {
            "metrics_initialized": len(self.metrics),
            "metrics_available": list(self.metrics.keys()),
            "embeddings_available": self.evaluator_embeddings is not None,
            "llm_type": type(self.evaluator_llm).__name__,
            "embeddings_type": type(self.evaluator_embeddings).__name__ if self.evaluator_embeddings else None
        }

# Backward compatibility alias
RAGAS_Eval_Pipeline = RAGASEvalPipeline


