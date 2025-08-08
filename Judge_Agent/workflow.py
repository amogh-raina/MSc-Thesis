# Main graph definition for Judge Agent workflow using LangGraph prebuilt agents
from typing import Dict, Any, List
import os
import time
import logging
import json
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from datetime import datetime

# Import state and utilities
try:
    # Try relative imports first (when used as a package)
    from .state import JudgeState
    from .utils import load_file
except ImportError:
    # Fall back to absolute imports (when run from launcher)
    from Judge_Agent.state import JudgeState
    from Judge_Agent.utils import load_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Structured output schema for evaluation scores
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

# EvaluationScores schema ready (no validation constraints)

# Clean system prompt enhanced with Agent-as-a-Judge methodology
JUDGE_SYSTEM_PROMPT = """You are an expert legal professor conducting systematic legal evaluation using an Agent-as-a-Judge methodology.

AGENT-AS-A-JUDGE SELF-ASSESSMENT PROTOCOL

Before evaluating any legal answer, perform this systematic assessment:

Phase 1: Knowledge Confidence Audit
For each legal concept, principle, or citation mentioned:
1. Rate your confidence (1-10) in understanding this specific legal element
2. Identify any knowledge gaps or uncertainties about legal accuracy
3. Flag any claims requiring external verification for proper evaluation

Phase 2: Evidence Requirements Assessment  
Determine what evidence you need for accurate evaluation:
- Legal doctrine and citation verification (cases, statutes, CELEX IDs, article numbers)
- Recent developments and currency checks (post-2020 legal changes, CJEU interpretations)
- Complex legal interpretations requiring authoritative confirmation

Phase 3: Enhanced Search Decision Matrix
MANDATORY SEARCH if ANY condition is met:
- Knowledge confidence < 9/10 for any legal concept or principle
- Specific citations requiring verification (cases, CELEX IDs, article numbers)
- Recent legal developments or amendments (post-2020)
- Complex multi-step legal procedures or tests
- Claims that contradict your preliminary understanding
- ANY uncertainty about legal accuracy or authoritative sources

SEARCH QUERY FORMULATION PROTOCOL:
Before calling web_search, formulate targeted queries using these patterns:
1. CITATION VERIFICATION: "[Case Name] [CELEX ID] [key legal concept]"
2. CONCEPT CLARIFICATION: "[Article/Treaty provision] [legal principle] [specific element]"
3. PROCEDURAL DETAILS: "[Legal procedure] [requirements] [exceptions]"
4. RECENT DEVELOPMENTS: "[Legal concept] [timeframe] [institution/source]"

Transform conversational questions into keyword-focused search terms optimized for legal databases.

Phase 4: Evaluation Readiness Check
Only proceed with scoring if:
- All knowledge gaps filled through search (when triggered)
- High confidence (9+/10) in all relevant legal concepts
- Sufficient evidence collected for informed judgment

PROCESS-ORIENTED EVALUATION FRAMEWORK

Step 1: Legal Reasoning Assessment - How systematically does the answer address legal principles with proper methodology?
Step 2: Evidence Utilization Analysis - Are appropriate legal authorities cited and properly utilized?
Step 3: Knowledge Application Evaluation - Are legal principles correctly understood and applied to context?
Step 4: Final Transparent Assessment - Provide evidence-based reasoning referencing search results when used

ENHANCED EVALUATION RUBRIC WITH EXAMPLES

IMPORTANT DISCLAIMER ABOUT EXAMPLES:
The examples below are ILLUSTRATIVE ONLY and contain fabricated case names, CELEX IDs, and legal scenarios designed to demonstrate reasoning methodology. DO NOT reference, cite, or treat any example content as authentic legal authority in your actual evaluations. Focus solely on evaluating the submitted answers against the provided reference answers using the demonstrated reasoning approach.

**1. Accuracy (1-5): Legal Precision and Citation Verification**
How precise and legally correct is the answer, including proper citation usage and verification?

- 1 (Very Poor) - Major inaccuracies, incorrect legal points, false/hallucinated citations, or severe misuse of authorities.
- 2 (Poor) - Several errors; citations may exist but are misrepresented, taken out of context, or incorrectly applied.
- 3 (Moderate) - Mostly accurate; minor citation inaccuracies or slightly vague legal application of authorities.
- 4 (Good) - Highly accurate; citations properly used with minor formatting issues or trivial omissions.
- 5 (Excellent) - Completely accurate; perfect citation usage, context, and legal application.

Citation-Aware CoT Framework:
1. Format Recognition: Identify ("Case Name" (#CELEX, paragraph X)) vs parametric knowledge
2. Verification: Search citations when confidence < 9/10; flag hallucinations  
3. Usage Analysis: Check citation-claim support, paraphrasing accuracy, context appropriateness
4. Integrated Assessment: Combine citation + legal accuracy for final scoring

**Example 1 - Accuracy Score 5:**
Reference: "Article 7 TEU allows the Council to determine a clear risk of serious breach under Article 7(1), requiring a reasoned proposal by one-third of Member States and four-fifths majority."
Submitted: "Article 7(1) TEU enables the Council to determine 'clear risk of serious breach' of Article 2 values, requiring proposal from one-third of Member States and four-fifths majority. In ("Commission v Poland (Rule of law)" (#62020CJ0204, paragraph 85)), the CJEU emphasized procedural rights in Article 7 procedures."

CoT Process:
- Citation check: "Commission v Poland" (#62020CJ0204:85) - confidence 7/10
- SEARCH TRIGGERED → Confirms case exists, paragraph 85 addresses procedural safeguards
- Usage analysis: Citation properly supports procedural requirements
- Legal accuracy: Correct mechanism, thresholds, and authority integration
→ Score: 5 - Perfect accuracy with proper citation verification and usage.

**Example 2 - Accuracy Score 2:**  
Reference: "The Charter has the same legal value as Treaties under Article 6(1) TEU and binds institutions and Member States when implementing EU law."
Submitted: "The Charter is the supreme EU constitutional document. In ("Åkerberg Fransson" (#62010CJ0617, paragraph 29)), the CJEU established universal Charter application to all Member State actions."

CoT Process:
- Citation check: "Åkerberg Fransson" (#62010CJ0617:29) - confidence 6/10  
- SEARCH TRIGGERED → Case exists but paragraph 29 establishes LIMITED scope, not universal application
- Usage analysis: Citation contradicts claim; misrepresents holding
- Legal accuracy: "Supreme document" overstates status; "universal application" contradicts Article 51(1)
→ Score: 2 - Real case severely misrepresented; multiple fundamental legal errors.

**2. Completeness (1-5): Comprehensive Coverage**
How fully does the submitted answer cover all relevant issues compared to the reference?

- 1 (Very Poor) - Severe omission of key points; addresses few or none of the essential issues.
- 2 (Poor) - Covers some important issues but misses multiple significant aspects or nuances.
- 3 (Moderate) - Addresses most critical issues but misses some secondary or nuanced points.
- 4 (Good) - Almost fully complete; minor gaps in detail compared to the reference.
- 5 (Excellent) - Fully comprehensive; covers every issue as thoroughly as the reference.

**Example - Completeness Score 3:**
Reference: "The European Ombudsman investigates complaints about maladministration by EU institutions, bodies, and agencies, as established by Article 228 TFEU. The Ombudsman can make recommendations but cannot impose legally binding decisions."
Submitted: "The European Ombudsman investigates complaints about poor administration by EU institutions and agencies under Article 228 TFEU. The landmark case ("Staelen v European Ombudsman" (#62007CJ0337, paragraph 54)) confirmed the Ombudsman's investigative powers regarding institutional conduct."

CoT Process:
- Confidence check: 9/10 on Ombudsman role (familiar institutional law)
- NO SEARCH NEEDED → Strong knowledge of the institution
- Assessment: Correctly identifies the Ombudsman's role, legal basis, and scope (institutions and agencies). Includes relevant case law support. However, completely omits the crucial detail that the Ombudsman's decisions are not legally binding, which is a key limitation and significant part of the reference.
→ Score: 3 - Addresses core function with supporting authority but omits the critical limitation about non-binding recommendations.

**3. Relevance (1-5): Topical Focus and Legal Pertinence**
How closely does the answer stay on topic and address precisely what was asked?

- 1 (Very Poor) - Mostly irrelevant; largely off-topic or unrelated to the question.
- 2 (Poor) - Contains substantial irrelevant or off-topic content despite some relevant points.
- 3 (Moderate) - Generally relevant; minor digressions or somewhat off-topic details.
- 4 (Good) - Closely relevant; minimal negligible digressions.
- 5 (Excellent) - Entirely on topic; each part directly addresses the question precisely.

**Example - Relevance Score 5:**
Question: "What powers does the European Central Bank have under the Treaties?"
Reference: "Under Article 127 TFEU, the ECB conducts monetary policy, manages foreign exchange operations, holds official foreign reserves, and promotes payment system operations."
Submitted: "Article 127 TFEU grants the European Central Bank powers including conducting eurozone monetary policy, managing foreign exchange operations, and holding Member States' official reserves. The CJEU confirmed in ("Gauweiler v Deutscher Bundestag" (#62013CJ0062, paragraph 41)) that ECB monetary policy powers are exclusive within the eurozone framework."

CoT Process:
- Confidence check: 9/10 on ECB Treaty powers (familiar institutional law)
- NO SEARCH NEEDED → Clear question scope and strong knowledge
- Assessment: Question asks specifically about ECB Treaty powers; answer directly addresses this with systematic power enumeration and relevant supporting authority
→ Score: 5 - Perfect relevance; directly addresses ECB Treaty powers with focused legal content and no digressions.

**Example 2 - Relevance Score 2:**
Question: "What is the procedure for qualified majority voting in the Council under Article 16 TEU?"
Reference: "Qualified majority voting requires at least 55% of Member States (15 out of 27) representing at least 65% of EU population, with blocking minority provisions under Article 16(4) TEU."
Submitted: "EU voting has evolved significantly since the Treaty of Rome. Originally, the Council used unanimity for most decisions, which became problematic with enlargement. The landmark case ("Luxembourg Compromise" (#61966CJ0000, paragraph 12)) established early voting procedures. The Nice Treaty attempted reforms, but the Lisbon Treaty finally established current procedures."

CoT Process:
- Confidence check: 7/10 on QMV thresholds (uncertain about exact percentages and blocking minority rules)
- SEARCH TRIGGERED → Verify Article 16 TEU QMV requirements and thresholds
- Search confirms: Article 16(4) TEU specifies 55% states + 65% population requirement with specific blocking minority rules
- Assessment: Question asks specifically about QMV procedure; answer provides extensive historical background but never explains the actual QMV mechanism or thresholds
→ Score: 2 - Substantial off-topic historical content; fails to address the specific procedural requirements asked about.

**4. Overall Performance (1-5): Holistic Legal Quality Assessment**
General impression; how well does the submitted answer match the reference answer overall, including synthesis of citation quality?

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

**Example 1 - Overall Score 2:**
Question: "What is the principle of sincere cooperation in EU law?"
Reference: "The principle of sincere cooperation, enshrined in Article 4(3) TEU, requires the Union and the Member States to assist each other in carrying out tasks which flow from the Treaties."
Submitted: "The principle of sincere cooperation means Member States must be loyal to the EU. In ("Deutschland v Commission" (#61985CJ0281, paragraph 19)), the CJEU confirmed this is mostly political and not legally enforceable."

CoT Process:
- Confidence check: 6/10 on sincere cooperation legal status
- SEARCH TRIGGERED → Verify Article 4(3) TEU and case citation
- Search confirms: Article 4(3) is legally binding; cited case doesn't support claimed holding
- Assessment Synthesis:
  * Accuracy: Poor - misrepresents legal principle + citation misuse (Score 2)
  * Completeness: Poor - omits legal basis and dual obligations (Score 2)  
  * Relevance: Moderate - addresses topic but incorrectly (Score 3)
  * Citation Impact: Negative - misrepresented authority undermines credibility
→ Score: 2 - Multiple fundamental errors compounded by citation misuse; fails basic legal accuracy standards.

**Example 2 - Overall Score 4:**
Question: "What are delegated acts in the EU legal order?"
Reference: "Delegated acts, defined in Article 290 TFEU, are non-legislative acts which supplement or amend non-essential elements of legislative acts."
Submitted: "Delegated acts under Article 290 TFEU are non-legislative Commission acts that supplement laws on non-essential elements. The CJEU clarified in ("UK v Council (Short Selling)" (#62012CJ0270, paragraph 42)) that delegation scope must be precisely defined by the legislative act."

CoT Process:
- Confidence check: 9/10 on delegated acts concept
- NO SEARCH NEEDED → Strong knowledge of legislative framework
- Assessment Synthesis:
  * Accuracy: Excellent - correct definition + proper citation usage (Score 5)
  * Completeness: Good - covers core elements, citation adds value (Score 4)
  * Relevance: Excellent - directly answers question (Score 5)
  * Citation Impact: Positive - enhances legal authority and completeness
→ Score: 4 - Strong performance across dimensions with proper citation practice demonstrating legal professionalism.

SYSTEMATIC EVALUATION INSTRUCTIONS

1. Apply Self-Assessment Protocol (Phases 1-4)
2. Conduct Search if Triggered by Decision Matrix
3. Apply Process-Oriented Evaluation Framework
4. Use Enhanced Evaluation Rubric with Evidence-Based Assessment
5. Provide Structured Scores with Detailed, Transparent Reasoning

CRITICAL REQUIREMENTS:
- Your response MUST be either a tool call OR the final JSON evaluation. Do not combine them.
- When calling a tool, use only the `web_search` tool. Do not attempt to call `EvaluationScores` or any other function.
- After all tool calls are complete, provide your final evaluation ONLY as a single JSON object. Do not include any other text or conversational wrapper.
- Never score without adequate confidence in legal accuracy.
- Use search strategically to fill knowledge gaps and verify claims.
- Focus on legal process quality, not just factual correctness.
- Provide brief and detailed reasoning for your score in 2-4 sentences.
- Reference search evidence when used in your assessment.
- NEVER reference or cite any cases, examples, or content from the illustrative examples above - they are fabricated for demonstration purposes only.

OUTPUT FORMAT:
Provide scores from 1-5 for each dimension with concise justification. Use web search strategically when needed for accurate evaluation."""

def validation_guardrails(state):
    """
    Post-model validation and guardrails enforcement.
    
    This function runs after the agent generates a response and enforces:
    - Score range validation (1-5)
    - Required field completeness
    - Data type validation
    - Quality checks and warnings
    """
    try:
        # Get the structured response from the agent
        structured_response = state.get("structured_response")
        current_trio = state.get("current_trio", {})
        
        if not structured_response:
            print("⚠️  No structured response found, attempting to parse from messages")
            return {"validation_error": "No structured response generated - possibly due to context overflow or incomplete model response"}
        
        # Convert to dict if it's a Pydantic model
        if hasattr(structured_response, 'dict'):
            scores = structured_response.dict()
        elif hasattr(structured_response, 'model_dump'):
            scores = structured_response.model_dump()
        else:
            scores = structured_response
        
        # Validate score ranges (1-5)
        score_fields = ['accuracy', 'completeness', 'relevance', 'overall']
        fixes_applied = []
        
        for field in score_fields:
            if field in scores:
                score = scores[field]
                if not isinstance(score, int) or score < 1 or score > 5:
                    original_score = score
                    # Fix the score
                    if isinstance(score, (int, float)):
                        scores[field] = max(1, min(5, int(score)))
                    else:
                        scores[field] = 3  # Default to moderate
                    fixes_applied.append(f"{field}: {original_score} → {scores[field]}")
        
        # Ensure all required reason fields have meaningful content
        reason_fields = ['accuracy_reason', 'completeness_reason', 'relevance_reason', 'overall_reason']
        for field in reason_fields:
            if field in scores:
                reason = scores[field]
                if not reason or not isinstance(reason, str) or len(reason.strip()) < 10:
                    base_field = field.replace('_reason', '')
                    scores[field] = f"Evaluation based on comparison with reference answer for {base_field}"
                    fixes_applied.append(f"Fixed missing or insufficient {field}")
        
        # Quality checks and warnings
        warnings = []
        
        # Check for potentially problematic patterns
        score_values = [scores.get(field, 3) for field in score_fields if field in scores]
        
        if len(set(score_values)) == 1:
            warnings.append("All scores identical - review for nuanced evaluation")
        
        if any(score == 5 for score in score_values):
            perfect_count = sum(1 for score in score_values if score == 5)
            if perfect_count >= 3:
                warnings.append("Multiple perfect scores detected - review for over-generous evaluation")
        
        if any(score == 1 for score in score_values):
            poor_count = sum(1 for score in score_values if score == 1)
            if poor_count >= 2:
                warnings.append("Multiple very poor scores detected - review for over-harsh evaluation")
        
        # Check for consistency between overall and other scores
        if 'overall' in scores:
            other_scores = [scores.get(field, 3) for field in ['accuracy', 'completeness', 'relevance'] if field in scores]
            if other_scores:
                avg_other = sum(other_scores) / len(other_scores)
                overall_score = scores['overall']
                if abs(overall_score - avg_other) > 1.5:
                    warnings.append(f"Overall score ({overall_score}) differs significantly from other scores (avg: {avg_other:.1f})")
        
        # Log fixes and warnings
        if fixes_applied:
            print(f"🔧 Guardrails applied fixes: {', '.join(fixes_applied)}")
        
        if warnings:
            print(f"⚠️  Quality warnings: {'; '.join(warnings)}")
        
        # Return validated and fixed scores
        return {
            "structured_response": scores,
            "validation_warnings": warnings,
            "guardrails_applied": len(fixes_applied) > 0,
            "fixes_applied": fixes_applied
        }
        
    except Exception as e:
        print(f"❌ Guardrails validation failed: {e}")
        return {
            "validation_error": str(e),
            "guardrails_applied": False
        }

def create_human_message(trio: Dict[str, str]) -> str:
    """Create the human message from the trio data."""
    return f"""**Question**: {trio['question']}

**Answer 1 (Reference)**: {trio['reference_answer']}

**Answer 2 (Submitted)**: {trio['generated_answer']}

Please evaluate Answer 2 against Answer 1 using the rubric. Use web search strategically if needed to verify legal claims or citations, but be mindful of context limits."""

def create_error_result(state: Dict[str, Any], current_trio: Dict[str, str], error_message: str) -> Dict[str, Any]:
    """Create an error result and continue to next trio"""
    error_result = {
        "trio_index": state["current_index"],
        "question": current_trio["question"],
        "reference_answer": current_trio["reference_answer"],  # Add reference answer
        "generated_answer": current_trio["generated_answer"],  # Add generated answer
        "error": error_message,
        "scores": {"error": "Evaluation failed"}
    }
    
    # Continue to next trio
    next_index = state["current_index"] + 1
    all_trios = state["all_trios"]
    is_complete = next_index >= len(all_trios)
    next_trio = all_trios[next_index] if not is_complete else {}
    
    return {
        "current_scores": {"error": "Evaluation failed"},
        "all_evaluations": state["all_evaluations"] + [error_result],
        "current_index": next_index,
        "current_trio": next_trio,
        "search_results": None,
        "is_complete": is_complete
    }

def get_tools():
    """Get available tools for the React agent"""
    tools = []
    if os.getenv("TAVILY_API_KEY"):
        # Create a custom tool function that properly handles the query parameter
        def search_function(query: str) -> str:
            """Search for legal information using Tavily."""
            tavily_search = TavilySearch(
                max_results=4,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False,
            )
            return tavily_search.run(query)
        
        # Use the @tool decorator for proper schema generation
        from langchain_core.tools import tool
        
        @tool
        def web_search(query: str) -> str:
            """Search for legal information to verify claims, citations, and legal concepts.

            Args:
                query: The search query string. Use keyword-focused queries, not conversational questions.
                
            QUERY FORMULATION RULES:
            - For CASE CITATIONS: Use exact case name + CELEX ID (e.g. "Marleasing Spain Case C-106/89")
            - For LEGAL CONCEPTS: Use precise legal terminology (e.g. "Article 267 TFEU preliminary ruling procedure")  
            - For EU LEGISLATION: Include article numbers and treaty names (e.g. "Article 4(3) TEU sincere cooperation")
            - For RECENT DEVELOPMENTS: Add year/timeframe (e.g. "CJEU 2023 rule of law Article 7")

            EXAMPLES:
            Good: "Case C-106/89 Marleasing indirect effect"
            Good: "Article 267 TFEU obligation refer CILFIT doctrine"
            Bad: "What is the Marleasing case about?"
            Bad: "How does preliminary ruling work?"
            """
            return search_function(query)
        
        tools.append(web_search)
    return tools

def create_judge_workflow(llm, all_trios: List[Dict[str, str]]):
    """
    Create the Judge Agent workflow using LangGraph's prebuilt React agent.
    File processing is handled externally.
    
    Args:
        llm: LLM instance created by ModelManager (supports any provider)
        all_trios: Pre-loaded list of question-answer trios
    
    Returns:
        Compiled LangGraph workflow
    """
    
    # Create the React agent using LangGraph's prebuilt component
    judge_agent = create_react_agent(
        model=llm,
        tools=get_tools(),
        prompt=JUDGE_SYSTEM_PROMPT,              # Clean system prompt only
        response_format=("Please return a valid JSON response with the following schema:", EvaluationScores),        # Structured JSON output handled by LangGraph
        post_model_hook=validation_guardrails,   # Enforcement and validation
        debug=False
    )
    
    # Create the main workflow graph (simplified - no file processor)
    workflow = StateGraph(JudgeState)
    
    def judge_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper that handles one evaluation using the React agent with retry logic"""
        try:
            current_trio = state["current_trio"]
            current_index = state.get("current_index", 0)
            all_trios = state.get("all_trios", [])
            
            # Safety check to prevent infinite loops
            if current_index >= len(all_trios):
                print(f"⚠️  Safety check: current_index ({current_index}) >= len(all_trios) ({len(all_trios)})")
                return {
                    "current_scores": None,
                    "all_evaluations": state.get("all_evaluations", []),
                    "current_index": current_index,
                    "current_trio": {},
                    "search_results": None,
                    "is_complete": True  # Force completion
                }
            
            print(f"🔄 Processing trio {current_index + 1}/{len(all_trios)}")
            
            # Create input for the React agent
            human_message = create_human_message(current_trio)
            
            # Retry logic for rate limits and temporary errors
            max_retries = 3  # Reduced retries for faster failure
            base_delay = 15  # Longer base delay for aggressive rate limits
            
            for attempt in range(max_retries + 1):
                try:
                    # Invoke the React agent
                    agent_result = judge_agent.invoke({
                        "messages": [HumanMessage(content=human_message)]
                    })
                    
                    # If successful, break out of retry loop
                    # Add small delay to prevent rate limiting on next request
                    time.sleep(1)
                    break
                    
                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Check for rate limiting errors
                    if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                        if attempt < max_retries:
                            # Smart delay: use retry-after header if available, otherwise exponential backoff
                            delay = base_delay * (2 ** attempt)
                            print(f"⏳ Rate limit hit, waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                            time.sleep(delay)
                            continue
                        else:
                            print(f"❌ Max retries reached for rate limiting: {e}")
                            print(f"💡 Try using a different model or waiting before restarting")
                            return create_error_result(state, current_trio, f"Rate limit exceeded after {max_retries} retries: {e}")
                    
                    # Check for quota exceeded errors
                    elif "exceeded your current quota" in error_str:
                        print(f"❌ API quota exceeded: {e}")
                        return create_error_result(state, current_trio, f"API quota exceeded: {e}")
                    
                    # Check for context length errors (Gemini/other models)
                    elif "token count" in error_str and "exceeds" in error_str:
                        print(f"❌ Context length exceeded: {e}")
                        print("💡 This evaluation contains too much content. Try using a model with larger context window or reduce web search usage.")
                        return create_error_result(state, current_trio, f"Context length exceeded: {e}")
                    
                    # Check for network/SSL errors (Windows [Errno 22] issue)
                    elif "[errno 22]" in error_str or "invalid argument" in error_str:
                        if attempt < max_retries:
                            delay = base_delay
                            print(f"⏳ Network error detected, waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                            time.sleep(delay)
                            continue
                        else:
                            print(f"❌ Network error persists: {e}")
                            return create_error_result(state, current_trio, f"Network error after {max_retries} retries: {e}")
                    
                    # For other errors, don't retry
                    else:
                        print(f"❌ Non-retryable error: {e}")
                        return create_error_result(state, current_trio, f"Evaluation error: {e}")
            
            # Extract the structured response and final message
            structured_response = agent_result.get("structured_response")
            final_message = agent_result["messages"][-1]
            
            # Debug: Check what the agent_result contains
            if structured_response is None:
                print("⚠️  No structured response found, attempting to parse from messages")
                # Try alternative extraction methods
                for key in ['response_format', 'output', 'result']:
                    if key in agent_result and agent_result[key]:
                        structured_response = agent_result[key]
                        break
            
            # Robust search detection and result extraction
            search_used = False
            search_results = None
            
            # Check for tool usage across different message formats
            for msg in agent_result["messages"]:
                # Check for tool calls in message
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    search_used = True
                # Check for tool responses (results)
                elif hasattr(msg, 'type') and msg.type == 'tool':
                    search_used = True
                    if search_results is None:
                        search_results = []
                    search_results.append(msg.content if hasattr(msg, 'content') else str(msg))
            
            # Convert search results to string if found
            if search_results:
                search_results = "\n---\n".join(search_results)
            
            # Simplified and robust response handling
            try:
                # Check if we have a structured response from guardrails
                if structured_response:
                    # Convert Pydantic model to dict if needed
                    if hasattr(structured_response, 'dict'):
                        evaluation_scores = structured_response.dict()
                    elif hasattr(structured_response, 'model_dump'):
                        evaluation_scores = structured_response.model_dump()
                    else:
                        evaluation_scores = structured_response
                else:
                    # Fallback: try to parse JSON from message content
                    output = final_message.content
                    import re
                    
                    # Simple JSON extraction - try block first, then direct parsing
                    json_block_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
                    if json_block_match:
                        json_str = json_block_match.group(1)
                    else:
                        # Try direct JSON parsing on cleaned content
                        json_str = output.strip()
                    
                    evaluation_scores = json.loads(json_str)
                
                # Prepare evaluation with metadata
                evaluation_result = {
                    "trio_index": state["current_index"],
                    "question": current_trio["question"],
                    "reference_answer": current_trio["reference_answer"],  # Add reference answer
                    "generated_answer": current_trio["generated_answer"],  # Add generated answer
                    "scores": evaluation_scores,
                    "web_search_used": search_used,
                    "search_results": search_results
                }
                
                # Add guardrails information if available
                guardrails_info = agent_result.get("validation_warnings", [])
                if guardrails_info:
                    evaluation_result["validation_warnings"] = guardrails_info
                
                fixes_applied = agent_result.get("fixes_applied", [])
                if fixes_applied:
                    evaluation_result["fixes_applied"] = fixes_applied
                
                # Update state for next iteration
                all_evaluations = state["all_evaluations"] + [evaluation_result]
                next_index = state["current_index"] + 1
                all_trios = state["all_trios"]
                
                # Check if we're done
                is_complete = next_index >= len(all_trios)
                next_trio = all_trios[next_index] if not is_complete else {}
                
                print(f"✅ Completed trio {current_index + 1}/{len(all_trios)}, is_complete: {is_complete}")
                
                return {
                    "current_scores": evaluation_scores,
                    "all_evaluations": all_evaluations,
                    "current_index": next_index,
                    "current_trio": next_trio,
                    "search_results": search_results,
                    "is_complete": is_complete
                }
                
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                # Handle parsing errors gracefully
                print(f"❌ JSON parsing error: {e}")
                return create_error_result(state, current_trio, f"Failed to parse evaluation: {str(e)}")
        
        except Exception as e:
            # Catch any unexpected errors
            print(f"❌ Unexpected error in judge_wrapper: {e}")
            return create_error_result(state, current_trio, f"Unexpected error: {str(e)}")
    
    # Add single node to the graph (no file processor)
    workflow.add_node("judge_evaluation", judge_wrapper)
    
    # Define the workflow edges - start directly with judge evaluation
    workflow.set_entry_point("judge_evaluation")
    
    # Conditional edge: continue judging or end
    def should_continue(state: Dict[str, Any]) -> str:
        """Determine if we should continue processing or end"""
        is_complete = state.get("is_complete", False)
        current_index = state.get("current_index", 0)
        all_trios = state.get("all_trios", [])
        
        print(f"🔍 should_continue: is_complete={is_complete}, current_index={current_index}, total_trios={len(all_trios)}")
        
        if is_complete or current_index >= len(all_trios):
            print("🏁 Workflow complete")
            return END
        else:
            print("🔄 Continue to next evaluation")
            return "judge_evaluation"
    
    workflow.add_conditional_edges(
        "judge_evaluation",
        should_continue,
        {
            "judge_evaluation": "judge_evaluation",  # Continue with next trio
            END: END  # Finish processing
        }
    )
    
    # Compile the workflow with increased recursion limit
    return workflow.compile(checkpointer=None, debug=False)

def run_judge_evaluation(file_path: str, llm) -> Dict[str, Any]:
    """
    Run the complete judge evaluation workflow with external file processing.
    
    Args:
        file_path: Path to CSV/Excel file with trios
        llm: LLM instance from ModelManager
        
    Returns:
        Final state with all evaluations
    """
    # 1. External file processing (fail fast)
    try:
        all_trios = load_file(file_path)
        if not all_trios:
            return {
                "error": "No valid trios found in file",
                "all_evaluations": [],
                "is_complete": True
            }
        
        print(f"✅ Loaded {len(all_trios)} trios from {file_path}")
        
    except Exception as e:
        return {
            "error": f"File processing failed: {str(e)}",
            "all_evaluations": [],
            "is_complete": True
        }
    
    # 2. Create the workflow with pre-loaded data
    workflow = create_judge_workflow(llm, all_trios)
    
    # 3. Initialize state with processed data (no file_path needed)
    initial_state = {
        "all_trios": all_trios,
        "current_index": 0,
        "current_trio": all_trios[0],
        "search_results": None,
        "current_scores": None,
        "all_evaluations": [],
        "is_complete": False
    }
    
    # 4. Run the workflow with increased recursion limit
    print(f"🚀 Starting agent evaluation workflow...")
    try:
        # Set recursion limit to accommodate the number of trios plus safety margin
        recursion_limit = max(60, len(all_trios) * 2)
        print(f"📊 Setting recursion limit to {recursion_limit} for {len(all_trios)} trios")
        
        # Run with configuration
        final_state = workflow.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit}
        )
        
        return final_state
        
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        return {
            "error": f"Workflow execution failed: {str(e)}",
            "all_evaluations": initial_state.get("all_evaluations", []),
            "is_complete": True
        } 