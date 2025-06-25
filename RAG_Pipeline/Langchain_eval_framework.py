from typing_extensions import Annotated, TypedDict
from typing import Dict, Any, Optional, List
import json
import re

### LANGCHAIN RAG EVALUATION FRAMEWORK
# 
# Note: This framework uses step-by-step reasoning prompts to improve evaluation quality.
# The explanations are generated internally but can be excluded from exports for zero-shot Q&A systems.
# The prompts encourage the model to "think" through the evaluation process for better accuracy.

class CorrectnessGrade(TypedDict):
    """Grade schema for correctness evaluation"""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    score: Annotated[int, ..., "Score on a scale of 1-5, where 1 is the lowest and 5 is the highest"]

class RelevanceGrade(TypedDict):
    """Grade schema for relevance evaluation"""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    score: Annotated[int, ..., "Score on a scale of 1-5, where 1 is the lowest and 5 is the highest"]

class GroundedGrade(TypedDict):
    """Grade schema for groundedness evaluation"""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    score: Annotated[int, ..., "Score on a scale of 1-5, where 1 is the lowest and 5 is the highest"]

class RetrievalRelevanceGrade(TypedDict):
    """Grade schema for retrieval relevance evaluation"""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    score: Annotated[int, ..., "Score on a scale of 1-5, where 1 is the lowest and 5 is the highest"]

class LangchainEvalPipeline:
    """
    LangChain RAG Evaluation Pipeline
    Evaluates RAG responses using four key metrics: correctness, relevance, groundedness, and retrieval relevance
    Based on: https://docs.smith.langchain.com/evaluation/tutorials/rag
    """
    
    def __init__(self, evaluator_llm):
        """
        Initialize the evaluation pipeline with an LLM evaluator
        
        Args:
            evaluator_llm: The LLM model to use for evaluation (e.g., ChatOpenAI, ChatGroq, etc.)
        """
        self.evaluator_llm = evaluator_llm
    
    def evaluate_correctness(self, question: str, reference_answer: str, generated_answer: str) -> Dict[str, Any]:
        """
        Evaluate correctness: How factually accurate is the generated answer compared to the reference?
        
        Args:
            question: The original question
            reference_answer: The ground truth answer
            generated_answer: The LLM-generated answer
            
        Returns:
            Dict containing explanation and score (1-5)
        """
        correctness_instructions = """You are a law teacher grading a quiz. 

        You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

        Here is the grade criteria to follow:
        (1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
        (2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness Scoring (1-5 scale):
- 5: Completely accurate, comprehensive, and well-grounded in legal sources
- 4: Mostly accurate with minor gaps or imprecisions
- 3: Generally accurate but missing some key points or has minor errors
- 2: Partially accurate but contains significant errors or omissions
- 1: Largely inaccurate or contains major factual errors

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.

Please respond in JSON format with 'explanation' and 'score' fields."""

        prompt_content = f"""QUESTION: {question}
        GROUND TRUTH ANSWER: {reference_answer}
        STUDENT ANSWER: {generated_answer}"""

        try:
            response = self.evaluator_llm.invoke([
                {"role": "system", "content": correctness_instructions}, 
                {"role": "user", "content": prompt_content}
            ])
            
            # Parse response - handle both string and structured responses
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Try to extract JSON from response
            result = self._parse_evaluation_response(response_text, default_score=3)
            return {
                "explanation": result.get("explanation", "No explanation provided"),
                "score": result.get("score", 3),
                "raw_response": response_text
            }
            
        except Exception as e:
            return {
                "explanation": f"Error during evaluation: {str(e)}",
                "score": 3,
                "error": str(e)
            }

    def evaluate_relevance(self, question: str, generated_answer: str) -> Dict[str, Any]:
        """
        Evaluate relevance: Does the generated answer address the question appropriately?
        
        Args:
            question: The original question
            generated_answer: The LLM-generated answer
            
        Returns:
            Dict containing explanation and score (1-5)
        """
        relevance_instructions = """You are a law teacher grading a quiz. 

        You will be given a QUESTION and a STUDENT ANSWER. 

        Here is the grade criteria to follow:
(1) Evaluate how well the STUDENT ANSWER addresses the QUESTION
(2) Consider whether the answer stays within the scope of the question
(3) Assess the level of detail and comprehensiveness in addressing the question
(4) Check if the answer provides useful information that helps answer the question

Relevance Scoring (1-5 scale):
- 5: Perfectly addresses the question with comprehensive, focused, and highly relevant information
- 4: Mostly addresses the question well with minor scope issues or slight lack of detail
- 3: Generally addresses the question but may have some scope drift or missing elements
- 2: Partially addresses the question with significant scope issues or limited relevance
- 1: Poorly addresses the question, largely off-topic, or provides irrelevant information

        Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.

Please respond in JSON format with 'explanation' and 'score' fields."""

        prompt_content = f"QUESTION: {question}\nSTUDENT ANSWER: {generated_answer}"
        
        try:
            response = self.evaluator_llm.invoke([
            {"role": "system", "content": relevance_instructions}, 
                {"role": "user", "content": prompt_content}
            ])
            
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            result = self._parse_evaluation_response(response_text, default_score=3)
            return {
                "explanation": result.get("explanation", "No explanation provided"),
                "score": result.get("score", 3),
                "raw_response": response_text
            }
            
        except Exception as e:
            return {
                "explanation": f"Error during evaluation: {str(e)}",
                "score": 3,
                "error": str(e)
            }

    def evaluate_groundedness(self, generated_answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluate groundedness: Is the generated answer supported by the retrieved context?
        
        Args:
            generated_answer: The LLM-generated answer
            context: The retrieved context/documents
            
        Returns:
            Dict containing explanation and score (1-5)
        """
        grounded_instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Evaluate how well the STUDENT ANSWER is grounded in the provided FACTS
(2) Assess the extent to which the answer relies on factual information from the context
(3) Check for any "hallucinated" information that goes beyond the provided facts
(4) Consider the balance between factual grounding and reasonable inference

Groundedness Scoring (1-5 scale):
- 5: Completely grounded in the facts with no hallucination, all claims directly supported
- 4: Mostly grounded with minor inferences that are reasonable and consistent with facts
- 3: Generally grounded but may include some reasonable inferences or minor gaps
- 2: Partially grounded with some unsupported claims or significant gaps in factual support
- 1: Poorly grounded with substantial hallucination or claims not supported by the facts

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.

Please respond in JSON format with 'explanation' and 'score' fields."""

        prompt_content = f"FACTS: {context}\nSTUDENT ANSWER: {generated_answer}"
        
        try:
            response = self.evaluator_llm.invoke([
                {"role": "system", "content": grounded_instructions}, 
                {"role": "user", "content": prompt_content}
            ])
            
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            result = self._parse_evaluation_response(response_text, default_score=3)
            return {
                "explanation": result.get("explanation", "No explanation provided"),
                "score": result.get("score", 3),
                "raw_response": response_text
            }
            
        except Exception as e:
            return {
                "explanation": f"Error during evaluation: {str(e)}",
                "score": 3,
                "error": str(e)
            }

    def evaluate_retrieval_relevance(self, question: str, context: str) -> Dict[str, Any]:
        """
        Evaluate retrieval relevance: Are the retrieved documents relevant to the question?
        
        Args:
            question: The original question
            context: The retrieved context/documents
            
        Returns:
            Dict containing explanation and score (1-5)
        """
        retrieval_relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student. 

Here is the grade criteria to follow:
(1) Evaluate how relevant the retrieved FACTS are to the QUESTION
(2) Assess the quality and comprehensiveness of the retrieved information
(3) Consider whether the facts contain key information needed to answer the question
(4) Check if the retrieval captured the most important aspects of the question

Retrieval Relevance Scoring (1-5 scale):
- 5: Perfectly relevant facts that comprehensively address all aspects of the question
- 4: Highly relevant facts that address most aspects of the question with minor gaps
- 3: Generally relevant facts that address some aspects but may miss key elements
- 2: Partially relevant facts with significant gaps or limited coverage of the question
- 1: Poorly relevant facts that largely miss the point or provide irrelevant information

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.

Please respond in JSON format with 'explanation' and 'score' fields."""

        prompt_content = f"FACTS: {context}\nQUESTION: {question}"
        
        try:
            response = self.evaluator_llm.invoke([
                {"role": "system", "content": retrieval_relevance_instructions}, 
                {"role": "user", "content": prompt_content}
            ])
            
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            result = self._parse_evaluation_response(response_text, default_score=3)
            return {
                "explanation": result.get("explanation", "No explanation provided"),
                "score": result.get("score", 3),
                "raw_response": response_text
            }
            
        except Exception as e:
            return {
                "explanation": f"Error during evaluation: {str(e)}",
                "score": 3,
                "error": str(e)
            }

    def evaluate_all(self, question: str, reference_answer: str, generated_answer: str, context: str) -> Dict[str, Any]:
        """
        Run all four RAG evaluation metrics
        
        Args:
            question: The original question
            reference_answer: The ground truth answer
            generated_answer: The LLM-generated answer
            context: The retrieved context/documents
            
        Returns:
            Dict containing all evaluation results
        """
        return {
            "correctness": self.evaluate_correctness(question, reference_answer, generated_answer),
            "relevance": self.evaluate_relevance(question, generated_answer),
            "groundedness": self.evaluate_groundedness(generated_answer, context),
            "retrieval_relevance": self.evaluate_retrieval_relevance(question, context)
        }

    def batch_evaluate(self, samples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of samples
        
        Args:
            samples: List of dicts, each containing 'question', 'reference_answer', 'generated_answer', 'context'
            
        Returns:
            List of evaluation results
        """
        results = []
        for i, sample in enumerate(samples):
            try:
                evaluation = self.evaluate_all(
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
                # Handle individual sample errors gracefully
                error_result = {
                    "sample_id": i,
                    "question": sample.get("question", ""),
                    "error": str(e),
                    "evaluation": {
                        "correctness": {"explanation": f"Error: {str(e)}", "score": 3},
                        "relevance": {"explanation": f"Error: {str(e)}", "score": 3},
                        "groundedness": {"explanation": f"Error: {str(e)}", "score": 3},
                        "retrieval_relevance": {"explanation": f"Error: {str(e)}", "score": 3}
                    }
                }
                results.append(error_result)
        
        return results

    def _parse_evaluation_response(self, response_text: str, **defaults) -> Dict[str, Any]:
        """
        Parse evaluation response, handling both JSON and plain text responses
        
        Args:
            response_text: The raw response from the LLM
            **defaults: Default values for different metrics
            
        Returns:
            Parsed evaluation dict
        """
        try:
            # Try to parse as JSON first
            if '{' in response_text and '}' in response_text:
                # Extract JSON part if embedded in text
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_part = response_text[start:end]
                return json.loads(json_part)
        except:
            pass
        
        # Fallback: try to extract values from text
        result = {}
        
        # Extract explanation (look for common patterns)
        if "explanation" in response_text.lower():
            lines = response_text.split('\n')
            explanation_lines = []
            capture = False
            for line in lines:
                if "explanation" in line.lower():
                    capture = True
                    explanation_lines.append(line.split(':', 1)[-1].strip())
                elif capture and line.strip():
                    explanation_lines.append(line.strip())
                elif capture and not line.strip():
                    break
            result["explanation"] = " ".join(explanation_lines) if explanation_lines else response_text
        else:
            result["explanation"] = response_text
        
        # Extract score (1-5) - all metrics now use scoring
        if "default_score" in defaults:
            score_match = re.search(r'score["\s:]*(\d)', response_text.lower())
            if score_match:
                score = int(score_match.group(1))
                # Ensure score is within valid range
                if 1 <= score <= 5:
                    result["score"] = score
                else:
                    result["score"] = defaults["default_score"]
            else:
                result["score"] = defaults["default_score"]
        
        return result

    def get_aggregate_scores(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate aggregate scores from a list of evaluation results
        
        Args:
            evaluation_results: List of results from batch_evaluate
            
        Returns:
            Dict with average scores for each metric
        """
        if not evaluation_results:
            return {}
        
        # Extract scores
        correctness_scores = []
        relevance_scores = []
        groundedness_scores = []
        retrieval_relevance_scores = []
        
        for result in evaluation_results:
            if "evaluation" in result:
                eval_data = result["evaluation"]
                
                # All metrics now use 1-5 scoring
                if "correctness" in eval_data and "score" in eval_data["correctness"]:
                    score = eval_data["correctness"]["score"]
                    correctness_scores.append(score)
                
                if "relevance" in eval_data and "score" in eval_data["relevance"]:
                    score = eval_data["relevance"]["score"]
                    relevance_scores.append(score)
                
                if "groundedness" in eval_data and "score" in eval_data["groundedness"]:
                    score = eval_data["groundedness"]["score"]
                    groundedness_scores.append(score)
                
                if "retrieval_relevance" in eval_data and "score" in eval_data["retrieval_relevance"]:
                    score = eval_data["retrieval_relevance"]["score"]
                    retrieval_relevance_scores.append(score)
        
        # Calculate averages
        aggregates = {}
        if correctness_scores:
            aggregates["avg_correctness"] = sum(correctness_scores) / len(correctness_scores)
        if relevance_scores:
            aggregates["avg_relevance"] = sum(relevance_scores) / len(relevance_scores)
        if groundedness_scores:
            aggregates["avg_groundedness"] = sum(groundedness_scores) / len(groundedness_scores)
        if retrieval_relevance_scores:
            aggregates["avg_retrieval_relevance"] = sum(retrieval_relevance_scores) / len(retrieval_relevance_scores)
        
        return aggregates

# Backward compatibility alias
Langchain_Eval_Pipeline = LangchainEvalPipeline
