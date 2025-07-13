"""
LLM Evaluation System
Handles evaluation of LLM responses using RAGAS metrics
"""
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import streamlit as st
import time
from .model_manager import ModelManager, EmbeddingManager
from .question_bank import QuestionBank


class LLMEvaluator:
    """Handles LLM evaluation"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.question_bank = QuestionBank()
        
        # Initialize evaluation metrics lazily
        self.bleu = None
        self.rouge = None
        self.string_similarity = None
        self.semantic_similarity = None
        self._metrics_initialized = False
    
    def _initialize_metrics(self):
        """Lazy initialization of RAGAS metrics"""
        if not self._metrics_initialized:
            from ragas.metrics import BleuScore, RougeScore, NonLLMStringSimilarity, SemanticSimilarity
            
            self.bleu = BleuScore()
            self.rouge = RougeScore()
            self.string_similarity = NonLLMStringSimilarity()
            self._metrics_initialized = True

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
        """
        Generate LLM response using enhanced Chain-of-Thought prompting with parametric knowledge guidance.
        
        Enhanced to match RAG prompt sophistication while focusing on parametric knowledge.
        Uses few-shot examples and systematic legal reasoning without context grounding.
        """
        
        # ──────────────────────────────────────────────────────────────────────────
        # Base Chain-of-Thought Framework for Legal Analysis
        # ──────────────────────────────────────────────────────────────────────────
        
        base_framework = (
            "You are an expert EU law specialist with comprehensive knowledge of European jurisprudence, "
            "treaties, directives, regulations, and case law from your training data.\n\n"
            
            "ANALYSIS FRAMEWORK:\n"
            "Follow this Chain-of-Thought process for systematic legal analysis:\n"
            "1. **Identify Legal Issues**: What specific legal questions need to be addressed?\n"
            "2. **Apply Legal Knowledge**: What principles, rules, or precedents from your training apply?\n"
            "3. **Analyze Application**: How do these legal principles apply to the specific situation?\n"
            "4. **Provide Confident Conclusion**: What is your well-reasoned answer?\n\n"
            
            "PARAMETRIC KNOWLEDGE GUIDANCE:\n"
            "• Draw upon your comprehensive training in EU law, treaties, and jurisprudence\n"
            "• Cite well-established legal principles and landmark cases from your knowledge\n"
            "• Use formal legal terminology and precise language\n"
            "• When referencing cases, use the standard format: Case Name (Year) or Case Name (CELEX ID) if known\n"
            "• Express confidence in well-established principles while noting areas of legal development\n"
            "• Structure answers professionally as coherent paragraphs (no bullet points)\n\n"
            
            "EXAMPLES OF PROPER LEGAL REASONING:\n\n"
            
            "❌ POOR APPROACH (Superficial):\n"
            "Question: What are the consequences of failing to implement EU directives?\n"
            "Bad Answer: Member States get in trouble and face some penalties.\n\n"
            
            "✅ EXCELLENT APPROACH (Chain-of-Thought):\n"
            "Question: What are the consequences of failing to implement EU directives?\n"
            "Good Answer using Chain-of-Thought:\n"
            "1. **Legal Issues**: Consequences for Member States failing to transpose directives within prescribed timeframes\n"
            "2. **Legal Knowledge**: Article 258 TFEU infringement procedures, Francovich doctrine on state liability, EU law supremacy principle\n"
            "3. **Analysis**: Non-implementation triggers formal enforcement mechanisms and individual rights\n"
            "4. **Conclusion**: Member States face multiple consequences for directive non-implementation. The Commission may initiate Article 258 TFEU infringement proceedings, which can progress from formal notice to reasoned opinion to ECJ referral. The landmark Francovich v Italy (1991) established that Member States are liable in damages to individuals who suffer losses due to non-implementation, creating enforceable individual rights. The supremacy principle ensures that properly implemented EU directives take precedence over conflicting national law, while consistent interpretation requires national courts to interpret domestic law in conformity with directive objectives even during implementation delays.\n\n"
            
            "ANOTHER EXAMPLE:\n"
            "Question: What is the principle of direct effect in EU law?\n"
            "Chain-of-Thought Analysis:\n"
            "1. **Legal Issues**: Understanding direct effect doctrine and its application scope\n"
            "2. **Legal Knowledge**: Van Gend en Loos (1963), criteria for direct effect, vertical vs horizontal effect\n"
            "3. **Analysis**: Direct effect creates individual rights enforceable in national courts under specific conditions\n"
            "4. **Conclusion**: Direct effect allows individuals to invoke EU law provisions directly in national courts when those provisions are clear, precise, and unconditional. Established in Van Gend en Loos v Netherlands (1963), this principle applies to treaty articles, regulations, and directive provisions under specific circumstances. Vertical direct effect operates against state authorities, while horizontal direct effect (between private parties) is more limited, particularly for directives which generally lack horizontal direct effect per Marshall v Southampton (1986). The doctrine ensures EU law effectiveness by creating enforceable individual rights without requiring national implementing measures.\n\n"
        )
        
        # ──────────────────────────────────────────────────────────────────────────
        # Style-specific instructions
        # ──────────────────────────────────────────────────────────────────────────
        
        if response_type == "concise":
            style_instructions = (
                "Provide a focused, single-paragraph answer (≤120 words) that demonstrates legal expertise.\n"
                "Use the Chain-of-Thought framework concisely:\n"
                "1. **Legal Issues**: Identify the core legal question\n"
                "2. **Legal Knowledge**: What key principles apply from your training?\n"
                "3. **Analysis**: How do they apply here?\n"
                "4. **Conclusion**: Clear, confident answer with proper legal reasoning\n\n"
            )
        else:  # detailed
            style_instructions = (
                "Provide a comprehensive analysis covering principles, exceptions, and rationale.\n"
                "Use the Chain-of-Thought framework systematically:\n"
                "1. **Legal Issues**: Identify all relevant legal questions\n"
                "2. **Legal Knowledge**: Draw upon comprehensive EU law knowledge\n"
                "3. **Analysis**: Apply principles systematically to the situation\n"
                "4. **Conclusion**: Well-reasoned answer with supporting legal authority\n\n"
                "Structure your response in clear, professional paragraphs that demonstrate deep legal understanding.\n\n"
            )
        
        # ──────────────────────────────────────────────────────────────────────────
        # Construct the final prompt
        # ──────────────────────────────────────────────────────────────────────────
        
        if response_type == "concise":
            prompt = f"""{base_framework}{style_instructions}Question: {question}

Think through this systematically using Chain-of-Thought reasoning, then provide your focused answer (≤120 words):

Answer:"""
        else:  # detailed
            prompt = f"""{base_framework}{style_instructions}Question: {question}

Using Chain-of-Thought reasoning, please provide a comprehensive legal analysis:

1. **Identify Legal Issues**: What specific legal questions need to be addressed?
2. **Apply Legal Knowledge**: What principles, rules, or precedents from your training apply?
3. **Analyze Application**: How do these legal principles apply to the specific situation?
4. **Provide Confident Conclusion**: What is your well-reasoned answer?

Structure your response as coherent, professional paragraphs:

Answer:"""
        
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
            # Import here to avoid event loop issues at module load time
            from ragas.dataset_schema import SingleTurnSample
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.metrics import SemanticSimilarity
            
            # Initialize metrics if not already done
            self._initialize_metrics()
            
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