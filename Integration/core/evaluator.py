"""
LLM Evaluation System
Handles evaluation of LLM responses using RAGAS metrics
"""
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import streamlit as st
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore, RougeScore, NonLLMStringSimilarity, SemanticSimilarity
from ragas.embeddings import LangchainEmbeddingsWrapper
import time
from core.model_manager import ModelManager, EmbeddingManager
from core.question_bank import QuestionBank


class LLMEvaluator:
    """Handles LLM evaluation"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.question_bank = QuestionBank()
        
        # Initialize evaluation metrics
        self.bleu = BleuScore()
        self.rouge = RougeScore()
        self.string_similarity = NonLLMStringSimilarity()
        self.semantic_similarity = None
    
    def setup_question_bank(self, data_dir: str, embedding_provider: str = None, 
                          embedding_model: str = None) -> bool:
        """Initialize question bank with optional embedding setup"""
        
        if embedding_provider and embedding_model:
            if not self.question_bank.setup_embedding_model(embedding_provider, embedding_model):
                st.warning("Failed to setup embedding model, will use fuzzy matching as fallback")
        
        return self.question_bank.load_questions_from_json(data_dir)
    
    def manual_evaluation(self, llm_provider: str, llm_model: str, 
                         question: str, reference_answer: str, response_type: str = "detailed") -> Dict[str, Any]:
        """Manual evaluation with provided reference answer"""
        
        llm = self.model_manager.create_llm(llm_provider, llm_model)
        if not llm:
            return {"error": "Failed to create LLM instance"}
        
        # Generate response
        generated_answer = self.generate_llm_response(llm, question, response_type)
        
        # Evaluate
        evaluation = asyncio.run(
            self.evaluate_single_response(generated_answer, reference_answer)
        )
        
        return {
            "question": question,
            "generated_answer": generated_answer,
            "reference_answer": reference_answer,
            "evaluation": evaluation,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "response_type": response_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def manual_evaluation_with_lookup(self, llm_provider: str, llm_model: str, 
                                    question: str, response_type: str = "detailed") -> Dict[str, Any]:
        """Manual evaluation with automatic reference lookup"""
        
        llm = self.model_manager.create_llm(llm_provider, llm_model)
        if not llm:
            return {"error": "Failed to create LLM instance"}
        
        # Find reference answer using embeddings (if available) or fuzzy matching
        reference_info = self.question_bank.find_reference_answer_embedding(question)
        reference_answer = None
        
        if reference_info and reference_info.get("question_data"):
            reference_answer = reference_info["question_data"]["answer_text"]
        elif reference_info and reference_info.get("answer_text"):
            reference_answer = reference_info["answer_text"]
        
        if not reference_answer:
            return {
                "error": "No reference answer found",
                "reference_lookup_info": reference_info,
                "suggestion": "Please provide reference answer manually"
            }
        
        # Generate response
        generated_answer = self.generate_llm_response(llm, question, response_type)
        
        # Evaluate
        evaluation = asyncio.run(
            self.evaluate_single_response(generated_answer, reference_answer)
        )
        
        return {
            "question": question,
            "generated_answer": generated_answer,
            "reference_answer": reference_answer,
            "evaluation": evaluation,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "response_type": response_type,
            "reference_lookup_info": reference_info,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_llm_response(self, llm, question: str, response_type: str = "detailed") -> str:
        """Generate LLM response without any context"""
        
        if response_type == "concise":
            prompt = f"""You are an assistant for legal question-answering tasks. Answer the following legal question concisely and directly using formal legal language:

            Question: {question}

            Provide a short and accurate response that highlights the central legal principle, using precise and professional terminology."""
                
        else:  # detailed
             prompt = f"""You are a legal expert assistant. Please answer the following legal question in a well-structured format using formal legal language:

            Question: {question}

            Your answer should be written in clear, paragraph form and must address the following:
            - The main legal principle or rule applicable to the situation
            - Relevant legal background, including statutory provisions, case law references, or precedents where appropriate
            - Any important exceptions, conditions, or contextual considerations

            Structure your response as coherent, informative paragraphs, ensuring clarity and legal accuracy. Avoid bullet points."""
        
        try:
            response = llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            error_msg = str(e)
            
            return f"Error generating response: {error_msg}"
    
    async def evaluate_single_response(self, generated_answer: str, reference_answer: str) -> Dict[str, float]:
        """Evaluate a single response using RAGAS metrics, including optional semantic similarity."""
        try:
            sample = SingleTurnSample(
                response=generated_answer,
                reference=reference_answer
            )

            # Always compute these metrics
            bleu_score = await self.bleu.single_turn_ascore(sample)
            rouge_score = await self.rouge.single_turn_ascore(sample)
            string_similarity_score = await self.string_similarity.single_turn_ascore(sample)

            results = {
                "bleu_score": float(bleu_score),
                "rouge_score": float(rouge_score),
                "string_similarity_score": float(string_similarity_score)
            }

            # Conditional semantic similarity
            if st.session_state.get("enable_emb_sem"):
                # Lazy‑init the metric the first time we need it
                if self.semantic_similarity is None:
                    provider = st.session_state.get("emb_provider_sem") or "HuggingFace"
                    model_id = st.session_state.get("emb_model_name_sem") or "sentence-transformers/all-MiniLM-L6-v2"

                    embedder = EmbeddingManager.create_embedding_model(provider, model_id)
                    self.semantic_similarity = SemanticSimilarity(
                        embeddings=LangchainEmbeddingsWrapper(embedder)
                    )

                sem_score = await self.semantic_similarity.single_turn_ascore(sample)
                results["semantic_similarity_score"] = float(sem_score)

            return results

        except Exception as e:  # noqa: BLE001
            st.error(f"Error during evaluation: {e}")
            # Zero‑fill expected keys so downstream code never KeyErrors
            keys = [
                "bleu_score",
                "rouge_score",
                "string_similarity_score",
            ]
            if st.session_state.get("enable_emb_sem"):
                keys.append("semantic_similarity_score")
            return {k: 0.0 for k in keys}


    
    def batch_evaluation(self, llm_provider: str, llm_model: str, 
                    max_questions: int = None, response_type: str = "detailed") -> List[Dict[str, Any]]:
        """Batch evaluation - directly uses database questions and answers (no embedding matching needed)"""
        
        llm = self.model_manager.create_llm(llm_provider, llm_model)
        if not llm:
            return []
        
        questions = self.question_bank.get_all_questions()
        if max_questions:
            questions = questions[:max_questions]
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, question_data in enumerate(questions):
            question_text = question_data["question_text"]
            reference_answer = question_data["answer_text"]  # Direct from database
            
            status_text.text(f"Processing question {idx + 1}/{len(questions)}: {question_text[:50]}...")
            
            try:
                # Generate response
                generated_answer = self.generate_llm_response(llm, question_text, response_type)
                
                # Evaluate directly without any embedding matching
                evaluation = asyncio.run(
                    self.evaluate_single_response(generated_answer, reference_answer)
                )
                
                result = {
                    "question_id": question_data["id"],
                    "year": question_data["year"],
                    "question_number": question_data["question_number"],
                    "question": question_text,
                    "generated_answer": generated_answer,
                    "reference_answer": reference_answer,
                    "evaluation": evaluation,
                    "llm_provider": llm_provider,
                    "llm_model": llm_model,
                    "response_type": response_type,
                    "source_file": question_data["source_file"],
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
                
                # Small delay to prevent rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                st.error(f"Error processing question {idx + 1}: {str(e)}")
                continue
            
            progress_bar.progress((idx + 1) / len(questions))
        
        status_text.text(f"Completed batch evaluation: {len(results)} questions processed")
        progress_bar.empty()
        
        return results
    
    def calculate_aggregate_scores(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate scores from batch results"""
        if not results:
            return {}

        totals: dict[str, float] = {}
        for res in results:
            for k, v in res["evaluation"].items():
                totals[k] = totals.get(k, 0.0) + v

        count = len(results)
        averages = {k: v / count for k, v in totals.items()}
        # Friendly display names ------------------------------------------------
        rename = {
            "bleu_score": "Avg BLEU",
            "rouge_score": "Avg ROUGE",
            "string_similarity_score": "Avg String Similarity",
            "semantic_similarity_score": "Avg Semantic Similarity",
        }
        return {rename.get(k, k): v for k, v in averages.items()}