from typing_extensions import Annotated, TypedDict
from typing import Dict, Any, Optional, List
import json
import re

### LANGCHAIN RAG EVALUATION FRAMEWORK - REFACTORED TO USE LLM JUDGE
# 
# Note: This framework now uses a single comprehensive LLM Judge approach instead of
# multiple separate evaluation prompts. Uses the complete judge prompt with few-shot examples.

class JudgeGrade(TypedDict):
    """Grade schema for LLM Judge evaluation"""
    accuracy: Annotated[int, ..., "Score on a scale of 1-5 for accuracy"]
    accuracy_reason: Annotated[str, ..., "Brief justification for accuracy score"]
    completeness: Annotated[int, ..., "Score on a scale of 1-5 for completeness"]
    completeness_reason: Annotated[str, ..., "Brief justification for completeness score"]
    relevance: Annotated[int, ..., "Score on a scale of 1-5 for relevance"]
    relevance_reason: Annotated[str, ..., "Brief justification for relevance score"]
    overall: Annotated[int, ..., "Score on a scale of 1-5 for overall performance"]
    overall_reason: Annotated[str, ..., "Brief justification for overall score"]

class LangchainEvalPipeline:
    """
    LangChain RAG Evaluation Pipeline - Refactored to use LLM Judge
    Evaluates RAG responses using a single comprehensive prompt with four key metrics:
    accuracy, completeness, relevance, and overall performance
    Based on legal domain-specific evaluation criteria with few-shot examples
    """
    
    def __init__(self, evaluator_llm):
        """
        Initialize the evaluation pipeline with an LLM evaluator
        
        Args:
            evaluator_llm: The LLM model to use for evaluation (e.g., ChatOpenAI, ChatGroq, etc.)
        """
        self.evaluator_llm = evaluator_llm
        self.judge_prompt = self._get_complete_judge_prompt()
    
    def _get_complete_judge_prompt(self) -> str:
        """
        Return the complete judge prompt exactly as specified in judge_prompt.md
        """
        return """## Prompt for Automated LLM Judge

### Persona / Role

You are an expert legal professor specialized in EU law with deep experience grading legal-theory assessments. Your task is to objectively evaluate the quality of short-answer responses by strictly adhering to the evaluation rubric. Provide precise, justified ratings.

---

### Task Description

You will evaluate answers to legal-theory questions. Each task shows:

- **Question**: The legal-theory question.
- **Answer 1 (Reference)**: The correct benchmark response (general legal reasoning, broad citations).
- **Answer 2 (Submitted)**: The response you will evaluate (explicit citations using case titles, CELEX IDs, paragraph numbers).

Evaluate Answer 2 by comparing it strictly against Answer 1 using the rubric dimensions defined below. Provide ratings from 1-5 for each dimension with a brief justification (one concise sentence per dimension).

Focus on the rubric criteria along with the writing style. Verify citations carefully if provided (check citation accuracy and relevance).

Output must follow the exact JSON schema provided at the end.

---

### Rubric Dimensions & Definitions

#### 1. Accuracy

How precise and legally correct is the answer compared to the reference?

- **1 (Very Poor)** - Major inaccuracies, incorrect legal points, or false/missing citations.
- **2 (Poor)** - Several errors; some points accurate but key legal elements or citations are substantially wrong.
- **3 (Moderate)** - Mostly accurate; minor inaccuracies or slightly vague legal phrasing.
- **4 (Good)** - Highly accurate; minor or trivial errors, no significant legal inaccuracies.
- **5 (Excellent)** - Completely accurate; perfectly matches all legal points and citations in the reference.

#### 2. Completeness

How fully does the submitted answer cover all relevant issues compared to the reference?

- **1 (Very Poor)** - Severe omission of key points; addresses few or none of the essential issues.
- **2 (Poor)** - Covers some important issues but misses multiple significant aspects or nuances.
- **3 (Moderate)** - Addresses most critical issues but misses some secondary or nuanced points.
- **4 (Good)** - Almost fully complete; minor gaps in detail compared to the reference.
- **5 (Excellent)** - Fully comprehensive; covers every issue as thoroughly as the reference.

#### 3. Relevance

How closely does the answer stay on topic and address precisely what was asked?

- **1 (Very Poor)** - Mostly irrelevant; largely off-topic or unrelated to the question.
- **2 (Poor)** - Contains substantial irrelevant or off-topic content despite some relevant points.
- **3 (Moderate)** - Generally relevant; minor digressions or somewhat off-topic details.
- **4 (Good)** - Closely relevant; minimal negligible digressions.
- **5 (Excellent)** - Entirely on topic; each part directly addresses the question precisely.

#### 4. Overall Performance

General impression; how well does the submitted answer match the reference answer overall?

- **1 (Very Poor)** - Clearly inadequate; significant inaccuracies, irrelevancies, and omissions.
- **2 (Poor)** - Below expectations; multiple noticeable deficiencies.
- **3 (Moderate)** - Adequate; broadly aligns but has clear errors or gaps.
- **4 (Good)** - Strong; closely aligns with the reference, minor and trivial deficiencies only.
- **5 (Excellent)** - Outstanding; matches the reference perfectly across all dimensions.

---

### Scenario-Based Few-shot Examples

> **Format note**\
> In each example the *Reference Answer* cites the legal authority **in a narrative way** (e.g. just the case name or doctrinal description), whereas the *Submitted Answer* follows the RAG style (case title in quotation marks + CELEX ID + optional paragraph number). Your job is to judge substance, not formatting; extra pinpoint citations in the Submitted Answer are welcome **as long as they are correct**.

**Example 1 - Excellent match**

- **Question**: Can EU Member States reserve jobs in the public administration exclusively for nationals?
- **Reference Answer** (benchmark): Posts may be restricted to nationals only if they involve the exercise of public authority or safeguarding the State's general interest as interpreted narrowly by the Court in its public-service case-law.
- **Submitted Answer** (to be scored): Under Article 45(4) TFEU, posts can be reserved for nationals solely where they involve the exercise of public authority. "Commission v Belgium" (#62001CJ0473, paragraph 39) confirms that purely administrative roles must remain open to all EU citizens, so a blanket nationality bar is unlawful.
- **Gold Ratings** (illustrative):
  - Accuracy 5 - The legal rule and citation are both correct.
  - Completeness 5 - Covers the narrow scope of the exception and its doctrinal source.
  - Relevance 5 - Fully on-point.
  - Overall 5 - Perfect alignment.

**Example 2 - Moderate match**

- **Question**: Do disproportionate language requirements in private employment breach EU free-movement rules?
- **Reference Answer**: Disproportionate or unnecessary language tests can amount to indirect discrimination contrary to Article 45 TFEU; legitimate requirements must be strictly proportionate.
- **Submitted Answer**: Language conditions are acceptable only if objectively justified. In "Groener" (#61987CJ0379, paragraph 19) the Court upheld a requirement linked to the job's teaching duties, showing that disproportionate tests would violate Article 45.
- **Gold Ratings** (illustrative):
  - Accuracy 4 - States the core rule accurately but gives only one example case and omits indirect-discrimination wording.
  - Completeness 3 - Misses the proportionality balancing and alternative-means test present in the reference.
  - Relevance 5 - Stays exactly on topic.
  - Overall 4 - Good but with notable gaps.

**Example 3 - Poor match**

- **Question**: Does moving from full-time to genuine part-time work strip an EU citizen of 'worker' status?
- **Reference Answer**: No. Part-time workers remain 'workers' so long as their activity is genuine and effective, not marginal or ancillary (Levin; Kempf).
- **Submitted Answer**: Switching to part-time automatically ends worker status under EU law, as shown in "Commission v Netherlands" (#62010CJ0542, paragraph 12).
- **Gold Ratings** (illustrative):
  - Accuracy 1 - Misstates the rule and cites an irrelevant case.
  - Completeness 1 - Omits the genuine-and-effective test entirely.
  - Relevance 2 - Mentions free movement but reaches the opposite conclusion.
  - Overall 1 - Substantially wrong.

---

### Explicit Guardrails / Rules

- **Do NOT penalise stylistic differences**. Extra precision in citations (e.g. CELEX IDs, paragraph numbers) is positive *if correct*.
- **Do NOT invent or assume information.** Judge only what is written.
- **If a Submitted citation looks fabricated or mismatched** (case does not support the proposition, wrong paragraph, non-existent CELEX), downgrade *Accuracy*.
- **Remain within the 1-5 integer scale** for every metric.
- **Output ONLY the JSON object - no prose, no markdown.**

---

### Output JSON Schema (strictly follow)

```json
{
  "accuracy": <int>,
  "accuracy_reason": "<brief justification sentence>",
  "completeness": <int>,
  "completeness_reason": "<brief justification sentence>",
  "relevance": <int>,
  "relevance_reason": "<brief justification sentence>",
  "overall": <int>,
  "overall_reason": "<brief justification sentence>"
}
```"""

    def evaluate_all(self, question: str, reference_answer: str, generated_answer: str, context: str = None) -> Dict[str, Any]:
        """
        Run comprehensive LLM Judge evaluation using single prompt
        
        Args:
            question: The original question
            reference_answer: The ground truth answer
            generated_answer: The LLM-generated answer
            context: The retrieved context (not used in judge approach but kept for compatibility)
            
        Returns:
            Dict containing all evaluation results in the same format as before
        """
        
        # Format the evaluation prompt with the specific answers
        evaluation_content = f"""**Question**: {question}

**Answer 1 (Reference)**: {reference_answer}

**Answer 2 (Submitted)**: {generated_answer}"""

        try:
            response = self.evaluator_llm.invoke([
                {"role": "system", "content": self.judge_prompt},
                {"role": "user", "content": evaluation_content}
            ])
            
            # Parse response - handle both string and structured responses
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Parse the JSON response
            judge_result = self._parse_judge_response(response_text)
            
            # Convert to the original format for backward compatibility
            return {
                "correctness": {
                    "explanation": judge_result.get("accuracy_reason", "No explanation provided"),
                    "score": judge_result.get("accuracy", 3),
                    "raw_response": response_text
                },
                "relevance": {
                    "explanation": judge_result.get("relevance_reason", "No explanation provided"),
                    "score": judge_result.get("relevance", 3),
                    "raw_response": response_text
                },
                "groundedness": {
                    "explanation": judge_result.get("completeness_reason", "No explanation provided"),
                    "score": judge_result.get("completeness", 3),
                    "raw_response": response_text
                },
                "retrieval_relevance": {
                    "explanation": judge_result.get("overall_reason", "No explanation provided"),
                    "score": judge_result.get("overall", 3),
                    "raw_response": response_text
                }
            }
            
        except Exception as e:
            return {
                "correctness": {"explanation": f"Error during evaluation: {str(e)}", "score": 3, "error": str(e)},
                "relevance": {"explanation": f"Error during evaluation: {str(e)}", "score": 3, "error": str(e)},
                "groundedness": {"explanation": f"Error during evaluation: {str(e)}", "score": 3, "error": str(e)},
                "retrieval_relevance": {"explanation": f"Error during evaluation: {str(e)}", "score": 3, "error": str(e)}
            }

    def judge_evaluate(self, question: str, reference_answer: str, generated_answer: str) -> Dict[str, Any]:
        """
        New method for direct judge evaluation (returns native judge format)
        
        Args:
            question: The original question
            reference_answer: The ground truth answer
            generated_answer: The LLM-generated answer
            
        Returns:
            Dict containing judge evaluation results in native format
        """
        
        # Format the evaluation prompt with the specific answers
        evaluation_content = f"""**Question**: {question}

**Answer 1 (Reference)**: {reference_answer}

**Answer 2 (Submitted)**: {generated_answer}"""

        try:
            response = self.evaluator_llm.invoke([
                {"role": "system", "content": self.judge_prompt},
                {"role": "user", "content": evaluation_content}
            ])
            
            # Parse response - handle both string and structured responses
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Parse the JSON response
            result = self._parse_judge_response(response_text)
            result["raw_response"] = response_text
            
            return result
            
        except Exception as e:
            return {
                "accuracy": 3,
                "accuracy_reason": f"Error during evaluation: {str(e)}",
                "completeness": 3,
                "completeness_reason": f"Error during evaluation: {str(e)}",
                "relevance": 3,
                "relevance_reason": f"Error during evaluation: {str(e)}",
                "overall": 3,
                "overall_reason": f"Error during evaluation: {str(e)}",
                "error": str(e),
                "raw_response": ""
            }

    def batch_evaluate(self, samples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of samples using LLM Judge
        
        Args:
            samples: List of dicts, each containing 'question', 'reference_answer', 'generated_answer'
            
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

    def _parse_judge_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM Judge response to extract scores and reasons
        
        Args:
            response_text: The raw response from the LLM
            
        Returns:
            Parsed evaluation dict with scores and reasons
        """
        try:
            # Try to parse as JSON first
            if '{' in response_text and '}' in response_text:
                # Extract JSON part if embedded in text
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_part = response_text[start:end]
                
                parsed = json.loads(json_part)
                
                # Validate that all required fields are present
                required_fields = [
                    "accuracy", "accuracy_reason",
                    "completeness", "completeness_reason", 
                    "relevance", "relevance_reason",
                    "overall", "overall_reason"
                ]
                
                if all(field in parsed for field in required_fields):
                    # Ensure scores are integers between 1-5
                    for score_field in ["accuracy", "completeness", "relevance", "overall"]:
                        if not isinstance(parsed[score_field], int) or not (1 <= parsed[score_field] <= 5):
                            parsed[score_field] = 3  # Default fallback
                    
                    return parsed
                    
        except json.JSONDecodeError:
            pass
        
        # Fallback: try to extract values from text using regex
        result = {}
        
        # Extract scores and reasons using regex patterns
        patterns = {
            "accuracy": r'"accuracy":\s*(\d+)',
            "accuracy_reason": r'"accuracy_reason":\s*"([^"]+)"',
            "completeness": r'"completeness":\s*(\d+)',
            "completeness_reason": r'"completeness_reason":\s*"([^"]+)"',
            "relevance": r'"relevance":\s*(\d+)',
            "relevance_reason": r'"relevance_reason":\s*"([^"]+)"',
            "overall": r'"overall":\s*(\d+)',
            "overall_reason": r'"overall_reason":\s*"([^"]+)"'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                if field.endswith("_reason"):
                    result[field] = match.group(1)
                else:
                    score = int(match.group(1))
                    result[field] = score if 1 <= score <= 5 else 3
            else:
                # Provide defaults for missing fields
                if field.endswith("_reason"):
                    result[field] = "Unable to parse reason from response"
                else:
                    result[field] = 3
        
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

    # Legacy methods preserved for backward compatibility but now call judge evaluation
    def evaluate_correctness(self, question: str, reference_answer: str, generated_answer: str) -> Dict[str, Any]:
        """Legacy method - now uses judge evaluation"""
        judge_result = self.judge_evaluate(question, reference_answer, generated_answer)
        return {
            "explanation": judge_result.get("accuracy_reason", "No explanation provided"),
            "score": judge_result.get("accuracy", 3),
            "raw_response": judge_result.get("raw_response", "")
        }

    def evaluate_relevance(self, question: str, generated_answer: str) -> Dict[str, Any]:
        """Legacy method - now uses judge evaluation (requires reference answer)"""
        # Note: This method now requires reference_answer, breaking change for legacy compatibility
        return {
            "explanation": "Legacy method - use judge_evaluate or evaluate_all instead",
            "score": 3,
            "error": "This method requires reference_answer in the new judge approach"
        }

    def evaluate_groundedness(self, generated_answer: str, context: str) -> Dict[str, Any]:
        """Legacy method - now uses judge evaluation (requires question and reference)"""
        # Note: This method now requires question and reference_answer, breaking change for legacy compatibility
        return {
            "explanation": "Legacy method - use judge_evaluate or evaluate_all instead",
            "score": 3,
            "error": "This method requires question and reference_answer in the new judge approach"
        }

    def evaluate_retrieval_relevance(self, question: str, context: str) -> Dict[str, Any]:
        """Legacy method - now uses judge evaluation (requires reference answer)"""
        # Note: This method now requires reference_answer, breaking change for legacy compatibility
        return {
            "explanation": "Legacy method - use judge_evaluate or evaluate_all instead",
            "score": 3,
            "error": "This method requires reference_answer in the new judge approach"
        }

# Backward compatibility alias
Langchain_Eval_Pipeline = LangchainEvalPipeline
