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
        Generate LLM response using optimized Chain-of-Thought prompting for parametric knowledge assessment.
        
        Designed to produce answers that include formal citations from the model's parametric knowledge,
        matching the citation style of the RAG system for fair comparison.
        """
        
        # ──────────────────────────────────────────────────────────────────────────
        # Role and Context Definition
        # ──────────────────────────────────────────────────────────────────────────
        
        role_context = (
            "You are an expert EU law specialist with comprehensive knowledge of European jurisprudence, "
            "treaties, directives, regulations, and case law from your training data."
        )
        
        # ──────────────────────────────────────────────────────────────────────────
        # Internal Reasoning Framework (Critical: Internal Only)
        # ──────────────────────────────────────────────────────────────────────────
        
        reasoning_framework = (
            "INTERNAL REASONING PROCESS:\n"
            "Think through this systematically using the following steps (DO NOT show these steps in your answer):\n"
            "1. Identify Legal Issues: What are the core legal questions in the user's query?\n"
            "2. Recall Relevant Law: What legal principles, treaties, and landmark cases from your knowledge are relevant? For each case, actively recall its name, its full CELEX ID, and its short case number if available.\n"
            "3. Apply Law to Facts: How do these principles and cases specifically answer the question?\n"
            "4. Construct Answer with Citations: Draft the final answer. When you mention a case, consult your recalled knowledge and select the most precise citation format you have available (Full CELEX > Short Case Number > Name only), following the output guidelines.\n\n"
            "Complete this reasoning process internally, then provide only your final answer."
        )
        
        # ──────────────────────────────────────────────────────────────────────────
        # Output Format and Citation Guidelines
        # ──────────────────────────────────────────────────────────────────────────
        
        output_guidelines = (
            "OUTPUT FORMAT AND CITATION GUIDELINES:\n"
            "Provide professional legal explanations using formal legal terminology.\n"
            "Reference treaty articles where relevant (e.g., 'Article 263 TFEU').\n"
            "When citing a case, you MUST follow this hierarchy of preference, using the most precise format you know:\n"
            "1. Level 1 (Best): Full CELEX ID -> \"Case Title\" (6-series CELEX ID).\n"
            "   Examples: \"Van Gend en Loos\" (61962CJ0026), \"Costa v ENEL\" (61964CJ0006).\n"
            "2. Level 2 (Good): Short Case Number -> \"Case Title\" (C-xxx/xx).\n"
            "   Examples: \"Marleasing\" (C-106/89), \"Internationale Handelsgesellschaft\" (C-11/70).\n"
            "3. Level 3 (Acceptable): Name only -> 'the Faccini Dori case', 'the Plaumann case'.\n"
            "Always strive for the most precise citation (Level 1 or 2). Only use Level 3 as a last resort.\n"
            "Structure answers as coherent, flowing paragraphs.\n"
            "Your answer should be based entirely on your parametric knowledge."
        )
        
        # ──────────────────────────────────────────────────────────────────────────
        # Style-specific instructions
        # ──────────────────────────────────────────────────────────────────────────
        
        if response_type == "concise":
            style_instruction = (
                "Provide a focused, single-paragraph answer (≤120 words) that demonstrates legal expertise and "
                "covers the essential legal principles and concepts."
            )
        else:  # detailed
            style_instruction = (
                "Provide a comprehensive legal analysis covering principles, exceptions, procedures, and rationale. "
                "Structure your response in clear, professional paragraphs that demonstrate deep legal understanding."
            )
        
        # ──────────────────────────────────────────────────────────────────────────
        # Few-Shot Examples Matching Reference Format
        # ──────────────────────────────────────────────────────────────────────────
        
        examples = (
            "EXAMPLES OF PROPER LEGAL EXPLANATIONS WITH CITATIONS:\n\n"
            "Example 1:\n"
            "Question: What is the significance of the 'Cassis de Dijon' ruling?\n"
            "Good Answer: The ruling in \"Rewe-Zentral AG v Bundesmonopolverwaltung für Branntwein\" (C-120/78), commonly known as 'Cassis de Dijon', is a cornerstone of EU single market law. It established the principle of mutual recognition, meaning that a product lawfully produced and marketed in one Member State should, in principle, be allowed in any other Member State. This applies even if the product does not comply with the technical or qualitative rules of the importing state. The ruling prevents Member States from imposing protectionist technical barriers to trade, unless necessary to satisfy mandatory requirements such as public health, which was a key consideration in the subsequent Torfaen case.\n\n"
            "Example 2:\n"
            "Question: How does EU law protect fundamental rights?\n"
            "Good Answer: The protection of fundamental rights in EU law has evolved significantly. Initially, the Court of Justice developed a body of case law, recognizing fundamental rights as general principles of EU law. A key ruling was \"Internationale Handelsgesellschaft\" (61970CJ0011), where the Court affirmed that respect for fundamental rights is an integral part of EU law. This was later codified in Article 6 of the Treaty on European Union and the legally binding Charter of Fundamental Rights of the European Union. Cases like \"Stauder v City of Ulm\" (C-29/69) were early steps in this jurisprudential path, ensuring that EU measures did not infringe upon fundamental human rights.\n\n"
            "Example 3:\n"
            "Question: What are the conditions for state liability under EU law?\n"
            "Good Answer: The principle of state liability allows individuals to claim compensation from a Member State for damages caused by a breach of EU law. The core conditions were established in \"Francovich and Bonifaci\" (61990CJ0006). They require that: first, the rule of law infringed must be intended to confer rights on individuals; second, the breach must be sufficiently serious; and third, there must be a direct causal link between the breach and the damage sustained. The Brasserie du Pêcheur and Factortame III cases further clarified that a breach is 'sufficiently serious' if a Member State has manifestly and gravely disregarded the limits on its discretion."
        )
        
        # ──────────────────────────────────────────────────────────────────────────
        # Construct Final Prompt
        # ──────────────────────────────────────────────────────────────────────────
        
        if response_type == "concise":
            prompt = f"""{role_context}

{reasoning_framework}

{output_guidelines}

{style_instruction}

{examples}

Question: {question}

Think through this systematically using internal reasoning, then provide your focused answer (≤120 words):

Answer:"""
        else:  # detailed
            prompt = f"""{role_context}

{reasoning_framework}

{output_guidelines}

{style_instruction}

{examples}

Question: {question}

Think through this systematically using internal reasoning, then provide your comprehensive legal analysis:

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