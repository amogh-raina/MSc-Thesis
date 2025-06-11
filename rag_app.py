import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
import time
import tempfile
import requests
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
import shutil  
import re

# Langchain Runnable imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain.schema import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field

# Model provider imports
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_nomic import NomicEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import JinaEmbeddings

# RAGAS imports
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore, RougeScore, NonLLMStringSimilarity, SemanticSimilarity
from ragas.embeddings import LangchainEmbeddingsWrapper
import importlib

# === RAG pipeline =========================================
from RAG_Pipeline.rag_pipeline import RAGPipeline
from RAG_Pipeline.title_index  import TitleIndex       # if you later need it in UI


# if 'rag_system' not in st.session_state:
#     st.session_state.rag_system = None
# if 'rag_evaluator' not in st.session_state:
#     st.session_state.rag_evaluator = None

if "rag_system" not in st.session_state:
    st.session_state.rag_system     = None     # holds RAGPipeline object
    st.session_state.rag_retriever  = None     # Chroma retriever for fast access
    st.session_state.rag_title_idx  = None     # TitleIndex for fuzzy CELEX lookup


try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import (SystemMessage as AzureSystem,
                                       UserMessage  as AzureUser,
                                       AssistantMessage as AzureAssistant)  # noqa: F401
    from azure.core.credentials import AzureKeyCredential
    AZURE_INFERENCE_AVAILABLE = True
except ImportError:
    # Keeps the rest of the app importable when azure-ai-inference isn't
    # installed (e.g. CI, first-time setup). ModelManager will raise a
    # clear error if a GitHub model is selected without the dependency.
    AZURE_INFERENCE_AVAILABLE = False
    


# Load environment variables
load_dotenv()

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class _GitHubChatCompletionsLLM(BaseChatModel):
    """Custom LLM for GitHub Models using Azure AI Inference"""
    
    _ENDPOINT = "https://models.github.ai/inference"
    
    # Pydantic fields for the model configuration
    model: str = Field(description="The model name to use")
    token: str = Field(description="GitHub token for authentication")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    top_p: float = Field(default=1.0, description="Top-p for generation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    
    # Private field to store the client
    _client: Optional[ChatCompletionsClient] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not AZURE_INFERENCE_AVAILABLE:
            raise ImportError("azure-ai-inference is required for GitHub models. pip install azure-ai-inference")
        
        # Initialize the client after validation
        self._client = ChatCompletionsClient(
            endpoint=self._ENDPOINT,
            credential=AzureKeyCredential(self.token),
        )

    @staticmethod
    def _to_azure(messages: List[BaseMessage]):
        """Convert LangChain messages → Azure AI messages."""
        out = []
        for m in messages:
            if isinstance(m, SystemMessage):
                out.append(AzureSystem(str(m.content)))
            elif isinstance(m, HumanMessage):
                out.append(AzureUser(str(m.content)))
            elif isinstance(m, AIMessage):
                out.append(AzureAssistant(str(m.content)))
            else:  # fallback
                out.append(AzureUser(str(m.content)))
        return out
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Required by BaseChatModel → returns ChatResult."""
        if self._client is None:
            raise RuntimeError("Client not initialized")
            
        azure_resp = self._client.complete(
            messages=self._to_azure(messages),
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        content = azure_resp.choices[0].message.content
        ai_msg = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation - naive wrapper for now"""
        return self._generate(messages, stop, run_manager, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "github-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

    def __repr__(self):
        return f"<_GitHubChatCompletionsLLM model={self.model!r}>"


class ModelManager:
    """Manages different model providers and their available models"""
    
    @staticmethod
    def get_available_llm_providers():
        """Get available LLM providers based on API keys"""
        providers = {}
        
        if os.getenv("NVIDIA_API_KEY"):
            providers["NVIDIA"] = {
                "models": [
                    "meta/llama-3.3-70b-instruct",
                    "microsoft/phi-4-mini-instruct",
                    "mistralai/mistral-large-2-instruct",
                    "microsoft/phi-3.5-moe-instruct",
                    "microsoft/phi-3-medium-128k-instruct",
                    "qwen/qwen2-7b-instruct"
                ]
            }
        
        if os.getenv("GROQ_API_KEY"):
            providers["Groq"] = {
                "models": [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant", 
                    "gemma2-9b-it"
                ]
            }
        
        if os.getenv("MISTRAL_API_KEY"):
            providers["Mistral"] = {
                "models": [
                    "mistral-small-2503",
                    "mistral-large-2411",
                    "ministral-3b-2410",
                    "ministral-8b-2410",
                    "open-mistral-7b",
                    "open-mixtral-8x7b",
                    "mistral-small-latest"
                ]
            }
        
        if os.getenv("OPENAI_API_KEY"):
            providers["OpenAI"] = {
                "models": [
                    "gpt-4-turbo-preview",
                    "gpt-4-1106-preview",
                    "gpt-3.5-turbo-1106",
                    "gpt-3.5-turbo"
                ]
            }
        
        if os.getenv("GOOGLE_API_KEY"):
            providers["Google"] = {
                "models": [
                    "gemini-2.0-flash",
                    "gemini-1.5-flash",
                    "gemini-2.0-flash-lite",
                    "gemini-2.5-flash-preview-05-20"
                ]
            }
        
        if os.getenv("GITHUB_TOKEN"):
            providers["Github"] = {
                "models": [
                    "meta/Llama-4-Scout-17B-16E-Instruct",
                    "Phi-4", 
                    "OpenAI o3",
                    "openai/gpt-4.1",
                    "openai/gpt-4o",
                    "deepseek/DeepSeek-R1",
                    "Grok-3"
                ]
            }

        
        return providers
    
    @staticmethod
    def create_llm(provider: str, model: str):
        """Create LLM instance based on provider and model"""
        try:
            if provider == "NVIDIA":
                return ChatNVIDIA(
                    model=model,
                    api_key=os.getenv("NVIDIA_API_KEY"),
                    temperature=0.1,
                    top_p=0.7
                )
            elif provider == "Groq":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found in environment variables")
                if not api_key.strip():
                    raise ValueError("GROQ_API_KEY is empty or contains only whitespace")
                
                return ChatGroq(
                    model=model,
                    api_key=api_key,
                    temperature=0.1,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                )
            elif provider == "Mistral":
                return ChatMistralAI(
                    model=model,
                    api_key=os.getenv("MISTRAL_API_KEY")
                )
            elif provider == "OpenAI":
                return ChatOpenAI(
                    model=model,
                    api_key=os.getenv("OPENAI_API_KEY")
                )

            elif provider == "Google":
                return ChatGoogleGenerativeAI(
                    model=model,
                    api_key=os.getenv("GOOGLE_API_KEY")
                )
            
            elif provider == "Github":
                token = os.getenv("GITHUB_TOKEN")
                if not token:
                    raise ValueError("GITHUB_TOKEN not found in environment variables")
                
                # UI model names -> registry IDs expected by Azure endpoint
                model_id_map = {
                    "OpenAI o3": "openai/o3",
                    "GPT‑4o (OpenAI)": "openai/gpt-4o",
                    "GPT‑4.1 (OpenAI)": "openai/gpt-4.1",
                    "Phi-4": "microsoft/Phi-4",
                    "meta/Llama-4-Scout-17B-16E-Instruct": "meta/Llama-4-Scout-17B-16E-Instruct",
                    "DeepSeek-R1": "deepseek/DeepSeek-R1",
                    "Grok-3": "xai/grok-3",
                }
                registry_model = model_id_map.get(model, model)
                
                return _GitHubChatCompletionsLLM(
                    model=registry_model,
                    token=token,
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=None,
                )
            
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except Exception as e:
            st.error(f"Error creating LLM {provider}/{model}: {str(e)}")
            return None

class SentenceTransformerEmbeddings:
    """Custom wrapper for SentenceTransformer embeddings"""
    
    def __init__(self, model_name: str, device: str = "cpu", **st_kwargs):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name, device=device, **st_kwargs)
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()
    

class EmbeddingManager:
    """Manages different embedding providers and models"""
    
    @staticmethod
    def get_available_embedding_providers():
        """Get available embedding providers based on API keys"""
        providers = {}
        
        if os.getenv("OPENAI_API_KEY"):
            providers["OpenAI"] = {
                "models": [
                    "text-embedding-3-large",
                    "text-embedding-3-small", 
                    "text-embedding-ada-002"
                ]
            }
        
        if os.getenv("NOMIC_API_KEY"):
            providers["Nomic"] = {
                "models": [
                    "nomic-embed-text-v1.5",
                    "nomic-embed-text-v1"
                ]
            }
        
        if os.getenv("JINA_API_KEY"):
            providers["JINA"] = {
                "models": [
                    "jina-embeddings-v3",
                    "jina-clip-v2",
                    "jina-embeddings-v2-base-en"
                ]
            }
            
        if os.getenv("GOOGLE_API_KEY"):
            providers["Google"] = {
                "models": [
                    "gemini-embedding-exp"
                ]
            }
        if os.getenv("MISTRAL_API_KEY"):
            providers["Mistral"] = {
                "models": [
                    "mistral-embed"
                ]
            }
        
        # HuggingFace models (no API key needed)
        providers["HuggingFace"] = {
            "models": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "BAAI/bge-large-en-v1.5",
                "BAAI/bge-base-en-v1.5"
            ]
        }

         # SentenceTransformers (direct usage)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            providers["SentenceTransformers"] = {
                "models": [
                    "all-MiniLM-L6-v2",
                    "all-mpnet-base-v2",
                    "paraphrase-multilingual-MiniLM-L12-v2",
                    "all-distilroberta-v1",
                    "multi-qa-mpnet-base-dot-v1",
                    "Qwen/Qwen3-Embedding-0.6B",
                    "nomic-ai/nomic-embed-text-v1.5"
                    #"legal-bert-base-uncased"  # Legal domain specific
                ]
            }
        
        return providers
    
    
    @staticmethod
    def create_embedding_model(provider: str, model: str):
        """Create embedding model instance based on provider and model"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if provider == "OpenAI":
                return OpenAIEmbeddings(
                    model=model,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            elif provider == "Nomic":
                return NomicEmbeddings(
                    model=model,
                    nomic_api_key=os.getenv("NOMIC_API_KEY")
                )
            elif provider == "JINA":
                try:
                    return JinaEmbeddings(
                        model_name=model,
                        jina_api_key=os.getenv("JINA_API_KEY")
                    )
                except Exception as e:
                    st.warning(f"Jina embedding setup failed: {e}")
                    raise
            
            elif provider == "Mistral":
                return MistralAIEmbeddings(
                    model=model,
                    mistral_api_key=os.getenv("MISTRAL_API_KEY")
                )
            
            elif provider == "Google":
                return GoogleGenerativeAIEmbeddings(
                    model=model,
                    api_key=os.getenv("GOOGLE_API_KEY")
                )
            
            elif provider == "HuggingFace":
                return HuggingFaceEmbeddings(
                    model_name=model,
                    model_kwargs={"device": device},  # Use 'cuda' if GPU available
                    encode_kwargs={'normalize_embeddings': True}
                )
            
             # ---------------------------------------------------------------
        # Sentence-Transformers provider
        # ---------------------------------------------------------------
            elif provider == "SentenceTransformers":
                if model == "nomic-ai/nomic-embed-text-v1.5":
                    emb = SentenceTransformerEmbeddings(
                        model, trust_remote_code=True, device=device
                    )
                else:
                    emb = SentenceTransformerEmbeddings(model, device=device)

                # ── async shim so ragas can call aembed_* gracefully ─────────
                if not hasattr(emb, "aembed_documents"):
                    async def _aembed_documents(texts):
                        return emb.embed_documents(texts)
                    emb.aembed_documents = _aembed_documents  # type: ignore[attr-defined]

                if not hasattr(emb, "aembed_query"):
                    async def _aembed_query(text):
                        return emb.embed_query(text)
                    emb.aembed_query = _aembed_query          # type: ignore[attr-defined]

                return emb

            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")

        except Exception as e:
            st.error(f"Error creating embedding model {provider}/{model}: {e}")
            return None



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
    

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="LLM Legal Knowledge Evaluator",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ LLM Legal Knowledge Evaluator")
    st.markdown("Evaluate LLM performance on legal questions")
    st.markdown("---")
    
    # Initialize evaluator
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = LLMEvaluator()
        st.session_state.evaluation_history = []
        st.session_state.question_bank_ready = False
        st.session_state.embedding_enabled = False

    # Sidebar configuration
    selected_llm_provider, selected_llm_model, response_type = sidebar_configuration()
    
    # Main content area
    if not st.session_state.question_bank_ready:
        st.info("👆 Please load the question bank from the sidebar to begin evaluation")
    else:
        if selected_llm_provider and selected_llm_model:
            tab1, tab2, tab3 = st.tabs([
            "📝 Manual Evaluation",
            "📊 Batch Evaluation",
            "🔍 RAG Q&A",
            ])
            with tab1:
                manual_evaluation_interface(selected_llm_provider, selected_llm_model, response_type)
            with tab2:
            #st.markdown("---")
                batch_evaluation_interface(selected_llm_provider, selected_llm_model, response_type)
            with tab3:
                rag_query_interface(selected_llm_provider,
                                    selected_llm_model,
                                    response_type)
            
        else:
            st.warning("Please select LLM provider and model from sidebar")


def sidebar_configuration():
    """Sidebar configuration"""
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data directory input
        st.subheader("📄 Question Database")
        data_dir = st.text_input(
            "Data Directory Path", 
            value="./JSON Trial 1",
            help="Path to directory containing BEUL_EXAM_*.json files"
        )
        
        # Embedding Configuration (only show if needed)
        st.subheader("🧠 Embedding Configuration")
        embedding_providers = EmbeddingManager.get_available_embedding_providers()

        # Manual evaluation embeddings
        use_embeddings_manual = st.checkbox(
            "Enable Embeddings (for manual evaluation only)",
            help="Embeddings are only used for manual question matching",
            key="enable_emb_manual"
        )

        # Semantic‑similarity embeddings
        use_embeddings_sem = st.checkbox(
            "Enable embeddings for semantic-similarity & RAG",
            help="Toggle to compute the semantic similarity metric",
            key="enable_emb_sem"
        )

        # — Manual matching model selector —
        selected_embedding_provider = None
        selected_embedding_model = None
        if use_embeddings_manual and embedding_providers:
            selected_embedding_provider = st.selectbox(
                "Embedding Provider (manual)",
                options=list(embedding_providers.keys()),
                key="emb_provider_manual"
            )
            if selected_embedding_provider:
                selected_embedding_model = st.selectbox(
                    "Embedding Model (manual)",
                    options=embedding_providers[selected_embedding_provider]["models"],
                    key="emb_model_manual"
                )

        # — Semantic similarity model selector —
        emb_provider_sem = None
        emb_model_name_sem = None
        if use_embeddings_sem and embedding_providers:
            emb_provider_sem = st.selectbox(
                "Embedding Provider (semantic & RAG)",
                options=list(embedding_providers.keys()),
                key="emb_provider_sem"
            )
            if emb_provider_sem:
                emb_model_name_sem = st.selectbox(
                    "Embedding Model (semantic & RAG)",
                    options=embedding_providers[emb_provider_sem]["models"],
                    key="emb_model_name_sem"
                )

        with st.sidebar.expander("📚 RAG Configuration"):
            st.markdown("Upload the **Paragraph-to-Paragraph** dataset (CSV or JSON) "
                        "and pick the *same* embedding model you enabled above.")

            # data
            uploaded_db = st.file_uploader("Dataset file", type=["csv", "json", "jsonl"])

            # persist dir
            chroma_dir  = st.text_input("Chroma directory", "./chroma_db")

            # build / rebuild buttons
            col_build, col_reset = st.columns(2)
            if col_build.button("🔧 Build / Load", use_container_width=True, key="rag_build_btn"):
                if not uploaded_db:
                    st.error("Please upload a CSV/JSON first.")
                elif not st.session_state.get("enable_emb_sem", False):
                    st.error("Please enable a **Semantic & RAG** embedding in the Embedding Configuration above.")
                else:
                    _setup_rag(uploaded_db, chroma_dir)

            if col_reset.button("🗑 Reset", use_container_width=True, key="rag_reset_btn"):
                _reset_rag(chroma_dir)


        # Question bank loading
        col1, col2 = st.columns(2)
        with col1:
            load_basic = st.button("Load Basic", help="Load without embeddings")
        with col2:
            load_enhanced = st.button("Load Enhanced", help="Load with embeddings")
        
        if load_basic or load_enhanced:
            with st.spinner("Loading question bank..."):
                if load_enhanced and use_embeddings_manual and selected_embedding_provider:
                    success = st.session_state.evaluator.setup_question_bank(
                        data_dir, selected_embedding_provider, selected_embedding_model
                    )
                    if success:
                        st.session_state.question_bank_ready = True
                        st.session_state.embedding_enabled = True
                        st.success("✅ Enhanced loading complete!")
                    else:
                        st.error("❌ Failed to load question bank")
                else:
                    success = st.session_state.evaluator.setup_question_bank(data_dir)
                    if success:
                        st.session_state.question_bank_ready = True
                        st.session_state.embedding_enabled = False
                        question_count = len(st.session_state.evaluator.question_bank.questions)
                        st.success(f"✅ Basic loading complete! Loaded {question_count} questions")
                    else:
                        st.error("❌ Failed to load question bank")
        
        st.markdown("---")
        
        # Model Selection
        st.subheader("🤖 Model Selection")
    

        # Continue with existing provider selection...
        llm_providers = st.session_state.evaluator.model_manager.get_available_llm_providers()
        
        if not llm_providers:
            st.error("❌ No LLM providers available. Please check your API keys.")
            return None, None, None
        
        selected_llm_provider = st.selectbox(
            "LLM Provider",
            options=list(llm_providers.keys()),
            help="Select the provider for your language model"
        )
        
        selected_llm_model = None
        if selected_llm_provider:
            selected_llm_model = st.selectbox(
                "LLM Model",
                options=llm_providers[selected_llm_provider]["models"],
                help="Select the specific model to use"
            )
        
        response_type = st.selectbox(
            "Response Type",
            ["detailed", "concise"],
            help="Choose whether to generate detailed or concise answers"
        )
        
        st.markdown("---")
        
        # Fixed Export section
        st.subheader("📊 Export Results")
        
        if st.session_state.evaluation_history:
            st.metric("Evaluations Completed", len(st.session_state.evaluation_history))
            
            export_format = st.selectbox("Export Format", ["excel", "json"])
            
            if st.button("📥 Export Results"):
                export_results(st.session_state.evaluation_history, export_format)
        else:
            st.text("No evaluations to export yet")
        
        return selected_llm_provider, selected_llm_model, response_type
        


def manual_evaluation_interface(selected_llm_provider, selected_llm_model, response_type):
    """Manual evaluation interface"""
    
    st.subheader("🔍 Manual Evaluation")
    
    question = st.text_area(
        "Enter your legal question:",
        placeholder="e.g., What is the principle of subsidiarity in EU law?",
        height=100
    )
    
    # Reference answer handling
    reference_mode = st.radio(
        "Reference Answer Source:",
        ["🤖 Auto-find from database", "✏️ Provide manually"],
        horizontal=True
    )
    
    if reference_mode == "🤖 Auto-find from database":
        if st.button("🚀 Generate & Evaluate with Auto-Reference"):
            if not question.strip():
                st.error("❌ Please provide a question")
            else:
                with st.spinner("🤖 Finding reference and evaluating..."):
                    result = st.session_state.evaluator.manual_evaluation_with_lookup(
                        selected_llm_provider, selected_llm_model, question, response_type
                    )
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                        if "suggestion" in result:
                            st.info(result["suggestion"])
                    else:
                        st.session_state.evaluation_history.append(result)
                        st.success("✅ Evaluation completed!")
                        show_evaluation_results(result)
    else:
        reference_answer = st.text_area(
            "Manual Reference Answer:",
            placeholder="Provide the expected/correct answer for evaluation",
            height=150
        )
        
        if st.button("🚀 Generate & Evaluate with Manual Reference"):
            if not question.strip():
                st.error("❌ Please provide a question")
            elif not reference_answer.strip():
                st.error("❌ Please provide a reference answer")
            else:
                with st.spinner("🤖 Generating and evaluating..."):
                    result = st.session_state.evaluator.manual_evaluation(
                        selected_llm_provider, selected_llm_model, question, reference_answer, response_type
                    )
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.session_state.evaluation_history.append(result)
                        st.success("✅ Evaluation completed!")
                        show_evaluation_results(result)

def show_evaluation_results(result):
    """Display detailed evaluation results"""
    
    st.markdown("---")
    st.subheader("📊 Detailed Evaluation Results")
    
    # Scores in columns
    col1, col2, col3, col4 = st.columns(4)
    evaluation = result["evaluation"]
    
    with col1:
        score = evaluation['bleu_score']
        color = "🟢" if score > 0.4 else "🟡" if score > 0.2 else "🔴"
        st.metric("BLEU Score", f"{score:.3f}")
        st.caption(f"{color} {'Excellent' if score > 0.4 else 'Good' if score > 0.2 else 'Needs improvement'}")

    with col2:
        score = evaluation['rouge_score']
        color = "🟢" if score > 0.4 else "🟡" if score > 0.2 else "🔴"
        st.metric("ROUGE Score", f"{score:.3f}")
        st.caption(f"{color} {'Excellent' if score > 0.4 else 'Good' if score > 0.2 else 'Needs improvement'}")

    with col3:
        score = evaluation['string_similarity_score']
        color = "🟢" if score > 0.6 else "🟡" if score > 0.4 else "🔴"
        st.metric("String Similarity", f"{score:.3f}")
        st.caption(f"{color} {'Excellent' if score > 0.6 else 'Good' if score > 0.4 else 'Needs improvement'}")

    with col4:
        if 'semantic_similarity_score' in evaluation:
            score = evaluation['semantic_similarity_score']
            color = "🟢" if score > 0.6 else "🟡" if score > 0.4 else "🔴"
            st.metric("Semantic Similarity", f"{score:.3f}")
            st.caption(f"{color} {'Excellent' if score > 0.6 else 'Good' if score > 0.4 else 'Needs improvement'}")
        else:
            st.metric("Semantic Similarity", "—")
    
    # Generated answer
    st.markdown("**🤖 Generated Answer:**")
    st.info(result["generated_answer"])
    
    # Reference lookup info
    if result.get("reference_lookup_info"):
        lookup_info = result["reference_lookup_info"]
        
        st.markdown("**🔍 Reference Matching Details:**")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.write(f"**Method:** {lookup_info.get('matching_method', 'N/A')}")
            st.write(f"**Model:** {lookup_info.get('model_used', 'N/A')}")
        
        with col_b:
            if lookup_info.get('similarity_score'):
                st.write(f"**Match Similarity:** {lookup_info['similarity_score']:.3f}")
            if lookup_info.get('question_data'):
                matched_data = lookup_info['question_data']
                st.write(f"**Source:** {matched_data.get('year', 'N/A')}-Q{matched_data.get('question_number', 'N/A')}")
    
    # Model details
    st.markdown("**⚙️ Configuration:**")
    provider_info = f"{result['llm_provider']}/{result['llm_model']}"
    st.write(f"LLM: {provider_info}")
    st.write(f"Response Type: {result['response_type']}")
    st.write(f"Timestamp: {result['timestamp']}")


def batch_evaluation_interface(selected_llm_provider, selected_llm_model, response_type):
    """Batch evaluation interface - simplified and cleaned"""
    
    st.subheader("🚀 Batch Evaluation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("Evaluate multiple questions automatically using database questions and answers")
        
        max_questions = st.number_input(
            "Maximum questions to evaluate:",
            min_value=1,
            max_value=100,
            value=10,
            help="Limit the number of questions for testing"
        )
        
        if st.button("🚀 Start Batch Evaluation", type="primary"):
            with st.spinner("Running batch evaluation..."):
                # Call the corrected batch evaluation method
                results = st.session_state.evaluator.batch_evaluation(
                    selected_llm_provider, selected_llm_model, max_questions, response_type
                )
                
                if results:
                    # Store results in session state
                    st.session_state.evaluation_history.extend(results)
                    
                    # Calculate aggregate scores
                    aggregate_scores = st.session_state.evaluator.calculate_aggregate_scores(results)
                    
                    st.success(f"✅ Batch evaluation completed! Processed {len(results)} questions")
                    
                    # Display aggregate metrics
                    st.subheader("📊 Results Summary")
                    col_a, col_b, col_c, col_d, col_e = st.columns(5)

                    with col_a:
                        st.metric("Total Questions", len(results))
                    with col_b:
                        st.metric("Avg BLEU", f"{aggregate_scores['Avg BLEU']:.3f}")
                    with col_c:
                        st.metric("Avg ROUGE", f"{aggregate_scores['Avg ROUGE']:.3f}")
                    with col_d:
                        st.metric("Avg String Similarity", f"{aggregate_scores['Avg String Similarity']:.3f}")
                    with col_e:
                        if 'Avg Semantic Similarity' in aggregate_scores:
                            st.metric("Avg Semantic Similarity", f"{aggregate_scores['Avg Semantic Similarity']:.3f}")
                    
                    # Results table preview
                    st.subheader("📋 Results Preview")
                    display_results_preview(results)
                    
                    # Export section
                    st.subheader("📥 Export Results")
                    col_export1, col_export2 = st.columns(2)
                    
                    with col_export1:
                        if st.button("Download Excel Report"):
                            export_results(results, "excel")
                    
                    with col_export2:
                        if st.button("Download JSON Report"):
                            export_results(results, "json")
                    
                    # Detailed analysis in expander
                    with st.expander("📊 Detailed Analysis", expanded=False):
                        display_detailed_analysis(results, selected_llm_provider, selected_llm_model, response_type)
                
                else:
                    st.error("❌ Batch evaluation failed - no results generated")
    
    with col2:
        display_batch_sidebar_info(selected_llm_provider, selected_llm_model)



def display_results_preview(results):
    """Display a preview table of results"""
    preview_data = []
    for i, result in enumerate(results[:10]):  # Show first 10
        preview_data.append({
        "Question #": i + 1,
        "Year": result.get("year", "N/A"),
        "Q#": result.get("question_number", "N/A"),
        "BLEU": f"{result['evaluation']['bleu_score']:.3f}",
        "ROUGE": f"{result['evaluation']['rouge_score']:.3f}",
        "Str Sim": f"{result['evaluation']['string_similarity_score']:.3f}",
        "Sem Sim": f"{result['evaluation'].get('semantic_similarity_score', float('nan')):.3f}",
        "Question Preview": (result["question"][:50] + "..." if len(result["question"]) > 50 else result["question"])
        })
    
    if preview_data:
        df_preview = pd.DataFrame(preview_data)
        st.dataframe(df_preview, use_container_width=True)
        
        if len(results) > 10:
            st.caption(f"Showing first 10 of {len(results)} results. Download full results using export buttons above.")


def display_detailed_analysis(results, llm_provider, llm_model, response_type):
    """Display detailed analysis of results"""
    
    # Score distribution analysis
    st.subheader("Score Distribution")
    
    scores_data = {
    "BLEU": [r["evaluation"]["bleu_score"] for r in results],
    "ROUGE": [r["evaluation"]["rouge_score"] for r in results],
    "String Sim": [r["evaluation"]["string_similarity_score"] for r in results],
    }
    if any('semantic_similarity_score' in r['evaluation'] for r in results):
        scores_data["Semantic Sim"] = [r["evaluation"].get("semantic_similarity_score", 0.0) for r in results]
    
    for metric, scores in scores_data.items():
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.write(f"**{metric} Score Statistics:**")
            st.write(f"Mean: {np.mean(scores):.3f}")
            st.write(f"Median: {np.median(scores):.3f}")
            st.write(f"Std Dev: {np.std(scores):.3f}")
        with col_dist2:
            st.write(f"Min: {np.min(scores):.3f}, Max: {np.max(scores):.3f}")
            
            # Score distribution
            high_scores = sum(1 for s in scores if s > 0.5)
            medium_scores = sum(1 for s in scores if 0.2 <= s <= 0.5)
            low_scores = sum(1 for s in scores if s < 0.2)
            
            st.write(f"High (>0.5): {high_scores}")
            st.write(f"Medium (0.2-0.5): {medium_scores}")
            st.write(f"Low (<0.2): {low_scores}")
    
    # Configuration summary
    st.subheader("⚙️ Configuration")
    st.write(f"**LLM Used:** {llm_provider}/{llm_model}")
    st.write(f"**Response Type:** {response_type}")
    st.write(f"**Questions Processed:** {len(results)}")
    st.write(f"**Evaluation Method:** Direct database matching (no embeddings)")


def display_batch_sidebar_info(selected_llm_provider, selected_llm_model):
    """Display sidebar info for batch evaluation"""
    st.subheader("⚙️ Batch Settings")
    
    # Statistics about available questions
    if st.session_state.question_bank_ready:
        total_questions = len(st.session_state.evaluator.question_bank.questions)
        st.metric("Available Questions", total_questions)
        st.info("🟢 Database Ready")
    
    # Tips
    st.markdown("---")
    st.subheader("💡 Tips")
    st.markdown("""
    **For Testing:**
    - Start with 5-10 questions
    - Check results quality
    
    **For Production:**
    - Use 50+ questions for statistics
    - Export results for analysis
    
    **Performance:**
    - ~2-3 seconds per question
    - Direct database matching
    """)
    
    # Current session results
    if st.session_state.evaluation_history:
        st.markdown("---")
        st.subheader("📊 Session Results")
        
        # Filter current model results
        current_results = [r for r in st.session_state.evaluation_history 
                          if r.get("llm_provider") == selected_llm_provider 
                          and r.get("llm_model") == selected_llm_model]
        
        if current_results:
            st.write(f"**Current Model:** {len(current_results)} evaluations")
            
            # Quick stats
            avg_scores = {
            'bleu': np.mean([r['evaluation']['bleu_score'] for r in current_results]),
            'rouge': np.mean([r['evaluation']['rouge_score'] for r in current_results]),
            'string': np.mean([r['evaluation']['string_similarity_score'] for r in current_results]),
        }
        if any('semantic_similarity_score' in r['evaluation'] for r in current_results):
            avg_scores['semantic'] = np.mean([r['evaluation'].get('semantic_similarity_score', 0.0) for r in current_results])
            
            st.write(f"Avg BLEU: {avg_scores['bleu']:.3f}")
            st.write(f"Avg ROUGE: {avg_scores['rouge']:.3f}")
            st.write(f"Avg String Sim: {avg_scores['string']:.3f}")
            if 'semantic' in avg_scores:
                st.write(f"Avg Semantic Sim: {avg_scores['semantic']:.3f}")
        
        st.write(f"**Total Session:** {len(st.session_state.evaluation_history)} evaluations")


# Improved export function
def export_results(results: List[Dict], format_type: str = "excel"):
    """Export results to Excel or JSON with proper error handling"""
    if not results:
        st.warning("No results to export")
        return
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "excel":
            # Prepare data for Excel export
            df_data = []
            for result in results:
                row = {
                    "timestamp": result.get("timestamp", ""),
                    "question_id": result.get("question_id", ""),
                    "year": result.get("year", ""),
                    "question_number": result.get("question_number", ""),
                    "llm_provider": result.get("llm_provider", ""),
                    "llm_model": result.get("llm_model", ""),
                    "response_type": result.get("response_type", ""),
                    "question": result.get("question", ""),
                    "generated_answer": result.get("generated_answer", ""),
                    "reference_answer": result.get("reference_answer", ""),
                    "bleu_score": result.get("evaluation", {}).get("bleu_score", 0),
                    "rouge_score": result.get("evaluation", {}).get("rouge_score", 0),
                    "string_similarity_score":    result.get("evaluation", {}).get("string_similarity_score", 0),
                    "semantic_similarity_score":  result.get("evaluation", {}).get("semantic_similarity_score", 0),
                    "source_file": result.get("source_file", "")
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            filename = f"llm_evaluation_results_{timestamp}.xlsx"
            
            # Create Excel file in memory
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Evaluation_Results')
            
            # Provide download button
            st.download_button(
                label="📥 Download Excel Report",
                data=output.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success(f"✅ Excel report ready for download: {filename}")
        
        else:  # JSON format
            filename = f"llm_evaluation_results_{timestamp}.json"
            json_data = json.dumps(results, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📥 Download JSON Report",
                data=json_data,
                file_name=filename,
                mime="application/json"
            )
            
            st.success(f"✅ JSON report ready for download: {filename}")
    
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")


def _setup_rag(file_obj, persist_dir):
    # 1⃣ stash file to a temp-path so Chroma can reopen it later
    tmp_path = Path(tempfile.gettempdir()) / f"tmp_{file_obj.name}"
    tmp_path.write_bytes(file_obj.getbuffer())

    # 2⃣ reuse the semantic-similarity embedding object already created
    provider  = st.session_state.get("emb_provider_sem")
    modelname = st.session_state.get("emb_model_name_sem")
    if not provider or not modelname:
        st.error("Semantic embedding not initialised.")
        return
    embed_obj = EmbeddingManager.create_embedding_model(provider, modelname)
    if embed_obj is None:
        return

    col_map = {
    "citing": {
        "text" : "TEXT_FROM",
        "celex": "CELEX_FROM",
        "para" : "NUMBER_FROM",
        "title": "TITLE_FROM",
    },
    "cited": {
        "text" : "TEXT_TO",
        "celex": "CELEX_TO",
        "para" : "NUMBER_TO",
        "title": "TITLE_TO",
    },
}
    # 3⃣ spin up / reuse the pipeline
    rag = RAGPipeline(
        dataset_path  = tmp_path,
        persist_dir   = Path(persist_dir),
        embedding     = embed_obj,
        k             = 4,               # or pull from a slider if you add one
        force_rebuild = False,
        col_map       = col_map,
    )
    with st.spinner("Building / loading vector store…"):
        rag.initialise()

    # 4⃣ store handles
    st.session_state.rag_system    = rag
    st.session_state.rag_retriever = rag.retriever
    st.session_state.rag_title_idx = rag.title_index
    st.success("✅ RAG ready!")

def _reset_rag(persist_dir):
    try:
        shutil.rmtree(persist_dir, ignore_errors=True)
    except Exception:
        pass
    for key in ("rag_system", "rag_retriever", "rag_title_idx"):
        st.session_state[key] = None
    st.info("RAG state cleared.")

def rag_query_interface(selected_provider: str,
                        selected_model: str,
                        response_type: str):
    if st.session_state.rag_system is None:
        st.info("Upload dataset and build RAG first.")
        return

    question = st.text_area("Legal question (free text):")
    if st.button("🪄 Generate", use_container_width=True) and question:
        llm = st.session_state.evaluator.model_manager.create_llm(
            selected_provider, selected_model
        )

        from langchain.chains import RetrievalQA
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.rag_retriever,
            chain_type_kwargs=dict(prompt=_rag_prompt(response_type)),
        )
        # Get both answer and context
        result = chain({"query": question, "return_source_documents": True})
        answer = result["result"]
        # You may need to extract the context string from the source documents:
        context = "\n".join([doc.page_content for doc in result.get("source_documents", [])])

        st.markdown("#### Answer")
        st.write(answer)

        # Validate citations
        invalid_cites = validate_citations(answer, context)
        if invalid_cites:
            st.warning(
                f"⚠️ The following citations in the answer are NOT present in the retrieved context: {invalid_cites}\n"
                "This may indicate hallucination or improper grounding."
            )
        else:
            st.success("✅ All citations in the answer are present in the context.")

def _rag_prompt(style):
    """
    Returns a PromptTemplate that instructs the LLM to only cite from the provided context,
    to avoid explicit statements about missing context, and to write answers as a single, well-structured paragraph.
    """
    if style == "detailed":
        tmpl = (
            "You are an EU-law specialist. Answer the following question using only the information provided in <context> below. "
            "Whenever you cite a legal passage, you MUST only cite paragraphs and CELEX IDs that appear in <context>. "
            "Do NOT invent or guess citations. If the context does not cover a point, answer as best you can without mentioning the absence of information. "
            "Write your answer in a coherent, well-structured manner, naturally incorporating relevant legal principles, background, and exceptions as appropriate. "
            "Cite using the format: (CELEX_ID:PARA_NO) — e.g. (62013CJ0196:113). "
            "If you quote or paraphrase, always cite immediately after the sentence."
            "\n\n<context>\n{context}\n\nQ: {question}\nA:"
        )
    else:
        tmpl = (
            "You are an EU-law expert. Provide a concise, authoritative answer using only the information in <context>. "
            "You MUST only cite paragraphs and CELEX IDs that appear in <context}. "
            "Do NOT invent or guess citations. If the context does not cover a point, answer as best you can without mentioning the absence of information. "
            "Write your answer as a single, clear paragraph. "
            "Cite using the format: (CELEX_ID:PARA_NO). Use a maximum of two citations unless strictly necessary."
            "\n\n<context>\n{context}\n\nQ: {question}\nA:"
        )
    from langchain.prompts import PromptTemplate
    return PromptTemplate(
        input_variables=["context", "question"],
        template=tmpl,
    )

def extract_citations(text):
    """
    Extracts all (CELEX_ID:PARA_NO) citations from the text.
    Returns a set of tuples: {(celex, para), ...}
    """
    pattern = r"\((\d{6,}CJ\d{4,}):(\d+)\)"
    return set(re.findall(pattern, text))

def context_citations(context):
    """
    Extracts all (CELEX_ID:PARA_NO) pairs from the context string.
    Returns a set of tuples: {(celex, para), ...}
    """
    return extract_citations(context)

def validate_citations(answer, context):
    """
    Checks if all citations in the answer are present in the context.
    Returns a list of invalid citations (those not found in context).
    """
    answer_cites = extract_citations(answer)
    context_cites = context_citations(context)
    invalid = answer_cites - context_cites
    return list(invalid)

if __name__ == "__main__":
    main()