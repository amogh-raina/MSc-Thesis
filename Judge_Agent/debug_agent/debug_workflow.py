# Debug workflow for LangGraph Studio testing
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import json
import os

# Add parent directories to path for imports
parent_dir = Path(__file__).parent.parent
project_root = parent_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import from main Judge Agent
from Judge_Agent.state import JudgeState
from Judge_Agent.workflow import (
    JUDGE_SYSTEM_PROMPT, 
    EvaluationScores, 
    validation_guardrails,
    get_tools,
    create_human_message
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

def create_studio_workflow():
    """Create an interactive workflow for LangGraph Studio testing"""
    
    # Create workflow that accepts human message input
    studio_workflow = StateGraph(JudgeState)
    
    def parse_trio_input(state):
        """Parse trio from human message input and initialize complete state"""
        messages = state.get("messages", [])
        if not messages:
            # Return complete state with error but all required fields
            return {
                "all_trios": [],
                "current_index": 0,
                "current_trio": {},
                "search_results": None,
                "current_scores": None,
                "all_evaluations": [],
                "is_complete": True,
                "error": "No input message provided. Please provide a trio in JSON format."
            }
        
        human_message = messages[-1]
        input_text = human_message.content if hasattr(human_message, 'content') else str(human_message)
        
        try:
            # Try to parse as JSON first
            trio_data = json.loads(input_text)
            
            # Validate required fields
            required_fields = ['question', 'reference_answer', 'generated_answer']
            if not all(field in trio_data for field in required_fields):
                return {
                    "all_trios": [],
                    "current_index": 0,
                    "current_trio": {},
                    "search_results": None,
                    "current_scores": None,
                    "all_evaluations": [],
                    "is_complete": True,
                    "error": f"Missing required fields. Need: {required_fields}"
                }
            
            trio = {
                'question': trio_data['question'],
                'reference_answer': trio_data['reference_answer'],
                'generated_answer': trio_data['generated_answer']
            }
            
        except json.JSONDecodeError:
            # If not JSON, try to parse as formatted text
            lines = input_text.strip().split('\n')
            trio = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('Question:'):
                    trio['question'] = line.replace('Question:', '').strip()
                elif line.startswith('Reference:'):
                    trio['reference_answer'] = line.replace('Reference:', '').strip()
                elif line.startswith('Generated:'):
                    trio['generated_answer'] = line.replace('Generated:', '').strip()
            
            # Validate we got all fields
            required_fields = ['question', 'reference_answer', 'generated_answer']
            if not all(field in trio and trio[field] for field in required_fields):
                return {
                    "all_trios": [],
                    "current_index": 0,
                    "current_trio": {},
                    "search_results": None,
                    "current_scores": None,
                    "all_evaluations": [],
                    "is_complete": True,
                    "error": "Could not parse trio. Use JSON format or:\nQuestion: <question>\nReference: <reference_answer>\nGenerated: <generated_answer>"
                }
        
        # Return complete state with all required fields initialized
        return {
            "all_trios": [trio],
            "current_index": 0,
            "current_trio": trio,
            "search_results": None,
            "current_scores": None,
            "all_evaluations": [],
            "is_complete": False
        }
    
    def studio_judge_evaluation(state):
        """Run actual judge evaluation using the real logic"""
        # Check for errors from parsing
        if state.get("error"):
            return state  # Pass through errors
        
        current_trio = state["current_trio"]
        
        # Create LLM instance - try multiple providers
        llm = None
        
        # Try to create LLM from available providers
        try:
            from Main.core.model_manager import ModelManager
            providers = ModelManager.get_available_llm_providers()
            
            # Try providers in order of preference
            provider_order = ["OpenAI", "NVIDIA", "Google", "Mistral", "Github"]
            
            for provider in provider_order:
                if provider in providers:
                    models = providers[provider]["models"]
                    if models:
                        llm = ModelManager.create_llm(provider, models[0])
                        if llm:
                            print(f"✅ Using {provider}/{models[0]} for Studio evaluation")
                            break
            
            if not llm:
                return {
                    **state,
                    "error": "No LLM available. Please check your API keys (OPENAI_API_KEY, NVIDIA_API_KEY, etc.)",
                    "is_complete": True
                }
                
        except Exception as e:
            return {
                **state,
                "error": f"Failed to create LLM: {str(e)}",
                "is_complete": True
            }
        
        # Create the React agent for this evaluation
        try:
            judge_agent = create_react_agent(
                model=llm,
                tools=get_tools(),
                prompt=JUDGE_SYSTEM_PROMPT,
                response_format=EvaluationScores,
                post_model_hook=validation_guardrails,
                debug=True  # Enable debug for Studio
            )
            
            # Create input for the React agent
            human_message = create_human_message(current_trio)
            
            # Invoke the React agent
            agent_result = judge_agent.invoke({
                "messages": [HumanMessage(content=human_message)]
            })
            
            # Extract results (similar to judge_wrapper logic)
            structured_response = agent_result.get("structured_response")
            final_message = agent_result["messages"][-1]
            
            # Check if web search was used
            search_used = any(
                hasattr(msg, 'tool_calls') and msg.tool_calls 
                for msg in agent_result["messages"]
            )
            
            # Extract search results if any
            search_results = None
            if search_used:
                tool_messages = [
                    msg for msg in agent_result["messages"] 
                    if hasattr(msg, 'type') and msg.type == 'tool'
                ]
                search_results = str(tool_messages) if tool_messages else None
            
            # Parse evaluation scores
            try:
                if structured_response:
                    evaluation_scores = structured_response.dict() if hasattr(structured_response, 'dict') else structured_response
                else:
                    # Fallback JSON parsing
                    output = final_message.content
                    import re
                    
                    json_block_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
                    if json_block_match:
                        json_str = json_block_match.group(1)
                    else:
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                        else:
                            json_str = output.strip()
                    
                    evaluation_scores = json.loads(json_str)
                
                # Create evaluation result
                evaluation_result = {
                    "trio_index": 0,
                    "question": current_trio["question"],
                    "scores": evaluation_scores,
                    "web_search_used": search_used,
                    "search_results": search_results,
                    "llm_provider": "Studio",
                    "llm_model": "Interactive",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Return complete state with all required fields
                return {
                    "all_trios": state["all_trios"],
                    "current_index": 1,
                    "current_trio": {},  # Empty since we're done
                    "search_results": search_results,
                    "current_scores": evaluation_scores,
                    "all_evaluations": [evaluation_result],
                    "is_complete": True,
                    "studio_result": evaluation_result  # Easy access for Studio
                }
                
            except Exception as parse_error:
                return {
                    **state,
                    "error": f"Failed to parse evaluation: {str(parse_error)}",
                    "raw_output": final_message.content if final_message else "No response",
                    "is_complete": True
                }
                
        except Exception as e:
            return {
                **state,
                "error": f"Evaluation failed: {str(e)}",
                "is_complete": True
            }
    
    # Add nodes to workflow
    studio_workflow.add_node("parse_input", parse_trio_input)
    studio_workflow.add_node("judge_evaluation", studio_judge_evaluation)
    
    # Set entry point
    studio_workflow.set_entry_point("parse_input")
    
    # Add edges
    def should_evaluate(state):
        """Determine if we should proceed to evaluation or end due to error"""
        if state.get("error"):
            return END
        return "judge_evaluation"
    
    studio_workflow.add_conditional_edges(
        "parse_input",
        should_evaluate,
        {
            "judge_evaluation": "judge_evaluation",
            END: END
        }
    )
    
    studio_workflow.add_edge("judge_evaluation", END)
    
    return studio_workflow.compile()

# Export workflow for LangGraph Studio
workflow = create_studio_workflow() 