"""
Question Bank Management
Handles loading and managing the question database
"""
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from core.model_manager import EmbeddingManager

class QuestionBank:
    """Manages the question database and reference answers"""
    
    def __init__(self):
        self.questions = []
        self.question_lookup = {}
        self.embedding_model = None
        self.question_embeddings = None
        self.question_texts = []
        self.embedding_provider = None
        self.embedding_model_name = None
    
    def setup_embedding_model(self, provider: str, model_name: str) -> bool:
        """Initialize embedding model"""
        try:
            self.embedding_model = EmbeddingManager.create_embedding_model(provider, model_name)
            if self.embedding_model:
                self.embedding_provider = provider
                self.embedding_model_name = model_name
                st.success(f"✅ Embedding model initialized: {provider}/{model_name}")
                return True
            return False
        except Exception as e:
            st.error(f"Failed to setup embedding model: {str(e)}")
            return False
    
    def load_questions_from_json(self, data_dir: str) -> bool:
        """Load questions and answers from JSON files"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            st.error(f"Data directory '{data_dir}' not found!")
            return False
        
        json_files = list(data_path.glob("BEUL_EXAM_*.json"))
        
        if not json_files:
            st.error(f"No BEUL_EXAM_*.json files found in '{data_dir}'")
            return False
        
        self.questions = []
        self.question_lookup = {}
        self.question_texts = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, json_path in enumerate(json_files):
            file_name = json_path.name
            status_text.text(f"Loading questions from {file_name}...")
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for record in data:
                    question_data = {
                        "id": record.get("id"),
                        "year": record.get("year"),
                        "question_number": record.get("question_number"),
                        "question_text": record.get("question_text", "").strip(),
                        "answer_text": record.get("answer_text", "").strip(),
                        "source_file": file_name
                    }
                    
                    self.questions.append(question_data)
                    self.question_texts.append(question_data["question_text"])
                    
                    if question_data["id"]:
                        self.question_lookup[str(question_data["id"])] = question_data
                
                progress_bar.progress((idx + 1) / len(json_files))
                
            except Exception as e:
                st.error(f"Error processing {file_name}: {str(e)}")
                return False
        
        status_text.text(f"Successfully loaded {len(self.questions)} questions")
        progress_bar.empty()
        
        # Generate embeddings if model is available
        if self.embedding_model:
            self._generate_embeddings()
        
        return True
    
    def _generate_embeddings(self):
        """Generate embeddings for all questions"""
        if not self.embedding_model or not self.question_texts:
            return
        
        st.info("🔄 Generating embeddings for questions...")
        progress_bar = st.progress(0)
        
        try:
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(self.question_texts), batch_size):
                batch = self.question_texts[i:i + batch_size]
                progress_bar.progress((i + len(batch)) / len(self.question_texts))
                
                if hasattr(self.embedding_model, 'embed_documents'):
                    batch_embeddings = self.embedding_model.embed_documents(batch)
                else:
                    batch_embeddings = [self.embedding_model.embed_query(text) for text in batch]
                
                all_embeddings.extend(batch_embeddings)
            
            self.question_embeddings = np.array(all_embeddings)
            st.success(f"✅ Generated embeddings for {len(self.question_texts)} questions")
            
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            self.question_embeddings = None
        finally:
            progress_bar.empty()
    
    def find_reference_answer_embedding(self, question_text: str, top_k: int = 3, 
                                      similarity_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """Find reference answer using embedding-based similarity"""
        if not self.embedding_model or self.question_embeddings is None:
            return self._find_reference_answer_fuzzy_with_info(question_text)
        
        try:
            if hasattr(self.embedding_model, 'embed_query'):
                query_embedding = self.embedding_model.embed_query(question_text)
            else:
                query_embedding = self.embedding_model.embed_documents([question_text])[0]
            
            query_embedding = np.array(query_embedding).reshape(1, -1)
            similarities = cosine_similarity(query_embedding, self.question_embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_similarities = similarities[top_indices]
            
            best_similarity = top_similarities[0]
            if best_similarity >= similarity_threshold:
                best_idx = top_indices[0]
                best_match = self.questions[best_idx]
                
                return {
                    "question_data": best_match,
                    "similarity_score": float(best_similarity),
                    "matching_method": "embedding",
                    "model_used": f"{self.embedding_provider}/{self.embedding_model_name}"
                }
            else:
                return {
                    "question_data": None,
                    "similarity_score": float(best_similarity),
                    "matching_method": "embedding",
                    "model_used": f"{self.embedding_provider}/{self.embedding_model_name}",
                    "message": f"Best match similarity ({best_similarity:.3f}) below threshold ({similarity_threshold})"
                }
                
        except Exception as e:
            st.error(f"Error in embedding matching: {str(e)}")
            return self._find_reference_answer_fuzzy_with_info(question_text)
    
    def find_reference_answer(self, question_text: str) -> Optional[str]:
        """Main interface for finding reference answers - uses embedding if available"""
        result = self.find_reference_answer_embedding(question_text)
        
        if result and result.get("question_data"):
            return result["question_data"]["answer_text"]
        
        return self._find_reference_answer_fuzzy(question_text)
    
    def _find_reference_answer_fuzzy(self, question_text: str) -> Optional[str]:
        """Original fuzzy matching as fallback"""
        normalized_question = self._normalize_text(question_text)
        
        best_match = None
        best_score = 0
        
        for question_data in self.questions:
            stored_question = self._normalize_text(question_data["question_text"])
            similarity = self._text_similarity(normalized_question, stored_question)
            if similarity > best_score and similarity > 0.8:
                best_score = similarity
                best_match = question_data
        
        return best_match["answer_text"] if best_match else None
    
    def _find_reference_answer_fuzzy_with_info(self, question_text: str) -> Dict[str, Any]:
        """Fuzzy matching with info structure for consistency"""
        answer = self._find_reference_answer_fuzzy(question_text)
        return {
            "question_data": None,
            "similarity_score": None,
            "matching_method": "fuzzy",
            "model_used": "fuzzy",
            "answer_text": answer,
        }
    
    def _normalize_text(self, text: str) -> str:
        """Enhanced text normalization for legal content"""
        import re
        
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[()[\]{}]', '', text)
        text = re.sub(r'art\.|article|art', 'article', text)
        text = re.sub(r'sec\.|section|sect', 'section', text)
        
        return text
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get_question_by_id(self, question_id: str) -> Optional[Dict]:
        """Get question by ID"""
        return self.question_lookup.get(str(question_id))
    
    def get_all_questions(self) -> List[Dict]:
        """Get all questions for batch processing"""
        return self.questions