# Simple LLM-as-Judge Evaluator - Pure Parametric Knowledge
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import re

class EvaluationScores(BaseModel):
    """Structured schema for legal evaluation scores"""
    accuracy: int = Field(description="Accuracy score from 1-5", ge=1, le=5)
    accuracy_reason: str = Field(description="Brief justification for accuracy score")
    completeness: int = Field(description="Completeness score from 1-5", ge=1, le=5)
    completeness_reason: str = Field(description="Brief justification for completeness score")
    relevance: int = Field(description="Relevance score from 1-5", ge=1, le=5)
    relevance_reason: str = Field(description="Brief justification for relevance score")
    overall: int = Field(description="Overall score from 1-5", ge=1, le=5)
    overall_reason: str = Field(description="Brief justification for overall score")

class LLMJudgeEvaluator:
    """
    Simple LLM-as-Judge Evaluator using only parametric knowledge.
    No web search, no agent capabilities - pure LLM evaluation.
    """
    
    def __init__(self, llm):
        """
        Initialize the LLM judge evaluator
        
        Args:
            llm: The LLM model to use for evaluation
        """
        self.llm = llm
        self.accuracy_prompt = self._get_accuracy_prompt()
        self.completeness_prompt = self._get_completeness_prompt()
        self.relevance_prompt = self._get_relevance_prompt()
        self.overall_prompt = self._get_overall_prompt()
    
    def _get_accuracy_prompt(self) -> str:
        """Accuracy evaluation prompt with citation-aware examples and COT reasoning"""
        return """You are an expert legal professor evaluating the accuracy of a legal answer.

IMPORTANT DISCLAIMER ABOUT EXAMPLES:
The examples below are ILLUSTRATIVE ONLY and contain fabricated case names, CELEX IDs, and legal scenarios designed to demonstrate reasoning methodology. DO NOT reference, cite, or treat any example content as authentic legal authority in your actual evaluations.

Task: Accuracy - How precise and legally correct is the answer, including proper citation usage and verification?
Rate the accuracy of Answer 2 compared to Answer 1 (reference) on a scale of 1-5.

Accuracy Scale:
- 1 (Very Poor) - Major inaccuracies, incorrect legal points, false/hallucinated citations, or severe misuse of authorities.
- 2 (Poor) - Several errors; citations may exist but are misrepresented, taken out of context, or incorrectly applied.
- 3 (Moderate) - Mostly accurate; minor citation inaccuracies or slightly vague legal application of authorities.
- 4 (Good) - Highly accurate; citations properly used with minor formatting issues or trivial omissions.
- 5 (Excellent) - Completely accurate; perfect citation usage, context, and legal application.

Citation-Aware CoT Framework:
1. Format Recognition: Identify ("Case Name" (#CELEX, paragraph X)) vs parametric knowledge
2. Verification: Assess citation existence and accuracy when confidence < 9/10
3. Usage Analysis: Check citation-claim support, paraphrasing accuracy, context appropriateness
4. Integrated Assessment: Combine citation + legal accuracy for final scoring

Example 1 - Score 5:
Reference: "Article 7 TEU allows the Council to determine a clear risk of serious breach under Article 7(1), requiring a reasoned proposal by one-third of Member States and four-fifths majority."
Submitted: "Article 7(1) TEU enables the Council to determine 'clear risk of serious breach' of Article 2 values, requiring proposal from one-third of Member States and four-fifths majority. In ("Commission v Poland (Rule of law)" (#62020CJ0204, paragraph 85)), the CJEU emphasized procedural rights in Article 7 procedures."

CoT Process:
- Citation check: "Commission v Poland" (#62020CJ0204:85) - properly formatted
- Usage analysis: Citation properly supports procedural requirements
- Legal accuracy: Correct mechanism, thresholds, and authority integration
→ Score: 5 - Perfect accuracy with proper citation verification and usage.

Example 2 - Score 2:  
Reference: "The Charter has the same legal value as Treaties under Article 6(1) TEU and binds institutions and Member States when implementing EU law."
Submitted: "The Charter is the supreme EU constitutional document. In ("Åkerberg Fransson" (#62010CJ0617, paragraph 29)), the CJEU established universal Charter application to all Member State actions."

CoT Process:
- Citation check: "Åkerberg Fransson" (#62010CJ0617:29) - properly formatted  
- Usage analysis: Citation contradicts claim; misrepresents holding
- Legal accuracy: "Supreme document" overstates status; "universal application" contradicts Article 51(1)
→ Score: 2 - Real case severely misrepresented; multiple fundamental legal errors.

Evaluation Instructions:
1. Legal Content Analysis: Compare all legal claims, principles, and citations for factual correctness
2. Citation Verification: Check accuracy of case names, article numbers, and legal authorities  
3. Usage Analysis: Assess if citations properly support claims and are contextually appropriate
4. Final Assessment: Provide score (1-5) and brief and detailed reasoning for your score in 2-4 sentences

NEVER reference or cite any cases, examples, or content from the illustrative examples above - they are fabricated for demonstration purposes only.

Output only JSON: {"accuracy": <score>, "accuracy_reason": "<brief explanation>"}"""

    def _get_completeness_prompt(self) -> str:
        """Completeness evaluation prompt with enhanced examples and COT reasoning"""
        return """You are an expert legal professor evaluating the completeness of a legal answer.

IMPORTANT DISCLAIMER ABOUT EXAMPLES:
The examples below are ILLUSTRATIVE ONLY and contain fabricated case names, CELEX IDs, and legal scenarios designed to demonstrate reasoning methodology. DO NOT reference, cite, or treat any example content as authentic legal authority in your actual evaluations.

Task: Completeness - How fully does the submitted answer cover all relevant issues compared to the reference?

Rate the completeness of Answer 2 to Answer 1 (reference) on a scale of 1-5.

Completeness Scale:
- 1 (Very Poor) - Severe omission of key points; addresses few or none of the essential issues.
- 2 (Poor) - Covers some important issues but misses multiple significant aspects or nuances.
- 3 (Moderate) - Addresses most critical issues but misses some secondary or nuanced points.
- 4 (Good) - Almost fully complete; minor gaps in detail compared to the reference.
- 5 (Excellent) - Fully comprehensive; covers every issue as thoroughly as the reference.

Examples with Chain-of-Thought Reasoning:

Example 1 - Score 3:
Reference: "The European Ombudsman investigates complaints about maladministration by EU institutions, bodies, and agencies, as established by Article 228 TFEU. The Ombudsman can make recommendations but cannot impose legally binding decisions."
Submitted: "The European Ombudsman investigates complaints about poor administration by EU institutions and agencies under Article 228 TFEU. The landmark case ("Staelen v European Ombudsman" (#62007CJ0337, paragraph 54)) confirmed the Ombudsman's investigative powers regarding institutional conduct."

CoT Analysis: Correctly identifies the Ombudsman's role, legal basis, and scope (institutions and agencies). Includes relevant case law support. However, completely omits the crucial detail that the Ombudsman's decisions are not legally binding, which is a key limitation and significant part of the reference.

→ Score: 3
→ Reasoning: Addresses core function with supporting authority but omits the critical limitation about non-binding recommendations.

Example 2 - Score 2:
Reference: "Under the Habitats Directive, any plan or project likely to have a significant effect on a Natura 2000 site must undergo an Appropriate Assessment. The project can only be authorised if it will not adversely affect the integrity of the site, unless there are no alternative solutions and imperative reasons of overriding public interest (IROPI)."
Submitted: "Projects that might affect a Natura 2000 site need a special assessment to check for environmental damage."

CoT Analysis: The submitted answer provides a very basic overview, mentioning the need for an assessment for projects affecting Natura 2000 sites. It completely omits the key legal concepts: the "Appropriate Assessment" terminology, the "site integrity" test, and the strict conditions for exceptions (no alternatives, IROPI). This is a significant gap in the procedural and substantive requirements.

→ Score: 2
→ Reasoning: Basic concept only; misses key legal tests like 'site integrity' and the IROPI exception criteria.

Evaluation Instructions:
1. Content Coverage Analysis: Identify all legal elements, principles, and components in both answers
2. Gap Identification: Note any missing legal concepts, authorities, or procedural elements
3. Comprehensiveness Assessment: Evaluate whether the answer addresses all aspects of the legal topic
4. Final Assessment: Provide score (1-5) and brief and detailed reasoning for your score in 2-4 sentences

NEVER reference or cite any cases, examples, or content from the illustrative examples above - they are fabricated for demonstration purposes only.

Output only JSON: {"completeness": <score>, "completeness_reason": "<brief explanation>"}"""

    def _get_relevance_prompt(self) -> str:
        """Relevance evaluation prompt with enhanced examples and COT reasoning"""
        return """You are an expert legal professor evaluating the relevance of a legal answer.

IMPORTANT DISCLAIMER ABOUT EXAMPLES:
The examples below are ILLUSTRATIVE ONLY and contain fabricated case names, CELEX IDs, and legal scenarios designed to demonstrate reasoning methodology. DO NOT reference, cite, or treat any example content as authentic legal authority in your actual evaluations.

Task: Relevance - How closely does the answer stay on topic and address precisely what was asked?
Rate the relevance of Answer 2 to Answer 1 (reference) on a scale of 1-5.

Relevance Scale:
- 1 (Very Poor) - Mostly irrelevant; largely off-topic or unrelated to the question.
- 2 (Poor) - Contains substantial irrelevant or off-topic content despite some relevant points.
- 3 (Moderate) - Generally relevant; minor digressions or somewhat off-topic details.
- 4 (Good) - Closely relevant; minimal negligible digressions.
- 5 (Excellent) - Entirely on topic; each part directly addresses the question precisely.

Examples with Chain-of-Thought Reasoning:

Example 1 - Score 5:
Question: "What powers does the European Central Bank have under the Treaties?"
Reference: "Under Article 127 TFEU, the ECB conducts monetary policy, manages foreign exchange operations, holds official foreign reserves, and promotes payment system operations."
Submitted: "Article 127 TFEU grants the European Central Bank powers including conducting eurozone monetary policy, managing foreign exchange operations, and holding Member States' official reserves. The CJEU confirmed in ("Gauweiler v Deutscher Bundestag" (#62013CJ0062, paragraph 41)) that ECB monetary policy powers are exclusive within the eurozone framework."

CoT Analysis: Question asks specifically about ECB Treaty powers; answer directly addresses this with systematic power enumeration and relevant supporting authority. All content focuses on ECB powers with no digressions.

→ Score: 5
→ Reasoning: Perfect relevance; directly addresses ECB Treaty powers with focused legal content and no digressions.

Example 2 - Score 2:
Question: "What is qualified majority voting in the Council of the EU?"
Reference: "Qualified majority voting requires at least 55% of Member States (15 out of 27) representing at least 65% of the EU population, as defined in Article 16(4) TEU."
Submitted: "EU decision-making has evolved significantly since the founding treaties. Originally, most decisions required unanimity, which became problematic as membership expanded. Various voting procedures exist including unanimity, qualified majority, and simple majority. The Council of the EU uses qualified majority voting for many areas, though some sensitive areas still require unanimity."

CoT Analysis: Question asks specifically about QMV procedure; answer provides extensive historical background but never explains the actual QMV mechanism or thresholds required.

→ Score: 2
→ Reasoning: Substantial off-topic historical content; fails to address the specific procedural requirements asked about.

Evaluation Instructions:
1. Question Focus Analysis: Identify exactly what the question is asking for
2. Content Relevance Assessment: Evaluate whether each part of the answer addresses the specific question
3. Digression Identification: Note any off-topic, tangential, or unnecessarily broad content
4. Final Assessment: Provide score (1-5) and brief and detailed reasoning for your score in 2-4 sentences

NEVER reference or cite any cases, examples, or content from the illustrative examples above - they are fabricated for demonstration purposes only.

Output only JSON: {"relevance": <score>, "relevance_reason": "<brief explanation>"}"""

    def _get_overall_prompt(self) -> str:
        """Overall performance evaluation prompt with enhanced examples and COT reasoning"""
        return """You are an expert legal professor providing an overall assessment of a legal answer.

IMPORTANT DISCLAIMER ABOUT EXAMPLES:
The examples below are ILLUSTRATIVE ONLY and contain fabricated case names, CELEX IDs, and legal scenarios designed to demonstrate reasoning methodology. DO NOT reference, cite, or treat any example content as authentic legal authority in your actual evaluations.

Task: Overall Performance - General impression; how well does the submitted answer match the reference answer overall, including synthesis of citation quality?

Rate the overall quality of Answer 2 compared to Answer 1 (reference) on a scale of 1-5.

Overall Performance Scale:
- 1 (Very Poor) - Clearly inadequate; significant inaccuracies, irrelevancies, omissions, or severe citation misuse.
- 2 (Poor) - Below expectations; multiple noticeable deficiencies including citation errors.
- 3 (Moderate) - Adequate; broadly aligns but has clear errors, gaps, or citation issues.
- 4 (Good) - Strong; closely aligns with reference, minor deficiencies, proper legal professionalism.
- 5 (Excellent) - Outstanding; matches reference perfectly with exemplary legal scholarship and citation practice.

Synthesis CoT Framework:
1. Integrate Parameter Findings: Synthesize accuracy, completeness, and relevance assessments
2. Citation Quality Impact: Consider citation verification findings from accuracy assessment
3. Legal Professionalism: Evaluate overall legal writing quality and authority usage
4. Holistic Judgment: Weight all factors for final assessment

Examples with Chain-of-Thought Reasoning:

Example 1 - Score 2:
Question: "What is the principle of sincere cooperation in EU law?"
Reference: "The principle of sincere cooperation, enshrined in Article 4(3) TEU, requires the Union and the Member States to assist each other in carrying out tasks which flow from the Treaties."
Submitted: "The principle of sincere cooperation means Member States must be loyal to the EU. In ("Deutschland v Commission" (#61985CJ0281, paragraph 19)), the CJEU confirmed this is mostly political and not legally enforceable."

CoT Analysis:
- Assessment Synthesis:
  * Accuracy: Poor - misrepresents legal principle + citation misuse (Score 2)
  * Completeness: Poor - omits legal basis and dual obligations (Score 2)  
  * Relevance: Moderate - addresses topic but incorrectly (Score 3)
  * Citation Impact: Negative - misrepresented authority undermines credibility
→ Score: 2 - Multiple fundamental errors compounded by citation misuse; fails basic legal accuracy standards.

Example 2 - Score 4:
Question: "What are delegated acts in the EU legal order?"
Reference: "Delegated acts, defined in Article 290 TFEU, are non-legislative acts which supplement or amend non-essential elements of legislative acts."
Submitted: "Delegated acts under Article 290 TFEU are non-legislative Commission acts that supplement laws on non-essential elements. The CJEU clarified in ("UK v Council (Short Selling)" (#62012CJ0270, paragraph 42)) that delegation scope must be precisely defined by the legislative act."

CoT Analysis:
- Assessment Synthesis:
  * Accuracy: Excellent - correct definition + proper citation usage (Score 5)
  * Completeness: Good - covers core elements, citation adds value (Score 4)
  * Relevance: Excellent - directly answers question (Score 5)
  * Citation Impact: Positive - enhances legal authority and completeness
→ Score: 4 - Strong performance across dimensions with proper citation practice demonstrating legal professionalism.

Evaluation Instructions:
1. Holistic Assessment: Consider accuracy, completeness, and relevance together
2. Citation Quality Impact: Factor in citation verification findings from accuracy assessment
3. Legal Professionalism: Assess overall legal writing quality and authority usage
4. Final Assessment: Provide score (1-5) and brief and detailed reasoning for your score in 2-4 sentences

NEVER reference or cite any cases, examples, or content from the illustrative examples above - they are fabricated for demonstration purposes only.

Output only JSON: {"overall": <score>, "overall_reason": "<brief explanation>"}"""

    def evaluate_accuracy(self, question: str, reference_answer: str, generated_answer: str) -> Dict[str, Any]:
        """Evaluate accuracy using LLM parametric knowledge"""
        evaluation_content = f"""Question: {question}

Answer 1 (Reference): {reference_answer}

Answer 2 (Submitted): {generated_answer}"""

        try:
            response = self.llm.invoke([
                {"role": "system", "content": self.accuracy_prompt},
                {"role": "user", "content": evaluation_content}
            ])
            
            result = self._parse_response(response)
            return {
                "score": result.get("accuracy", 3),
                "explanation": result.get("accuracy_reason", "No explanation provided"),
                "raw_response": self._get_response_content(response)
            }
        except Exception as e:
            return {
                "score": 3,
                "explanation": f"Error during accuracy evaluation: {str(e)}",
                "error": str(e)
            }

    def evaluate_completeness(self, question: str, reference_answer: str, generated_answer: str) -> Dict[str, Any]:
        """Evaluate completeness using LLM parametric knowledge"""
        evaluation_content = f"""Question: {question}

Answer 1 (Reference): {reference_answer}

Answer 2 (Submitted): {generated_answer}"""

        try:
            response = self.llm.invoke([
                {"role": "system", "content": self.completeness_prompt},
                {"role": "user", "content": evaluation_content}
            ])
            
            result = self._parse_response(response)
            return {
                "score": result.get("completeness", 3),
                "explanation": result.get("completeness_reason", "No explanation provided"),
                "raw_response": self._get_response_content(response)
            }
        except Exception as e:
            return {
                "score": 3,
                "explanation": f"Error during completeness evaluation: {str(e)}",
                "error": str(e)
            }

    def evaluate_relevance(self, question: str, reference_answer: str, generated_answer: str) -> Dict[str, Any]:
        """Evaluate relevance using LLM parametric knowledge"""
        evaluation_content = f"""Question: {question}

Answer 1 (Reference): {reference_answer}

Answer 2 (Submitted): {generated_answer}"""

        try:
            response = self.llm.invoke([
                {"role": "system", "content": self.relevance_prompt},
                {"role": "user", "content": evaluation_content}
            ])
            
            result = self._parse_response(response)
            return {
                "score": result.get("relevance", 3),
                "explanation": result.get("relevance_reason", "No explanation provided"),
                "raw_response": self._get_response_content(response)
            }
        except Exception as e:
            return {
                "score": 3,
                "explanation": f"Error during relevance evaluation: {str(e)}",
                "error": str(e)
            }

    def evaluate_overall(self, question: str, reference_answer: str, generated_answer: str) -> Dict[str, Any]:
        """Evaluate overall performance using LLM parametric knowledge"""
        evaluation_content = f"""Question: {question}

Answer 1 (Reference): {reference_answer}

Answer 2 (Submitted): {generated_answer}"""

        try:
            response = self.llm.invoke([
                {"role": "system", "content": self.overall_prompt},
                {"role": "user", "content": evaluation_content}
            ])
            
            result = self._parse_response(response)
            return {
                "score": result.get("overall", 3),
                "explanation": result.get("overall_reason", "No explanation provided"),
                "raw_response": self._get_response_content(response)
            }
        except Exception as e:
            return {
                "score": 3,
                "explanation": f"Error during overall evaluation: {str(e)}",
                "error": str(e)
            }

    def evaluate_all(self, question: str, reference_answer: str, generated_answer: str) -> Dict[str, Any]:
        """Run all evaluations and return results in the same format as agent judge"""
        accuracy_result = self.evaluate_accuracy(question, reference_answer, generated_answer)
        completeness_result = self.evaluate_completeness(question, reference_answer, generated_answer)
        relevance_result = self.evaluate_relevance(question, reference_answer, generated_answer)
        overall_result = self.evaluate_overall(question, reference_answer, generated_answer)
        
        # Return in the same format as the agent judge for compatibility
        return {
            "correctness": accuracy_result,  # Map accuracy to correctness for compatibility
            "groundedness": completeness_result,  # Map completeness to groundedness
            "relevance": relevance_result,
            "retrieval_relevance": overall_result  # Map overall to retrieval_relevance
        }

    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse LLM response to extract scores and reasons"""
        response_text = self._get_response_content(response)
        
        try:
            # Try to parse as JSON first
            if '{' in response_text and '}' in response_text:
                # Extract JSON part if embedded in text
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_part = response_text[start:end]
                
                parsed = json.loads(json_part)
                
                # Ensure scores are integers between 1-5
                for score_field in ["accuracy", "completeness", "relevance", "overall"]:
                    if score_field in parsed:
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

    def _get_response_content(self, response) -> str:
        """Extract content from response object"""
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response) 