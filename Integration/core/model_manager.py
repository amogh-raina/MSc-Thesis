"""
Model and Embedding Management
Handles LLM and embedding model instantiation
"""
import os
import torch
from typing import Dict, List, Optional
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_nomic import NomicEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import JinaEmbeddings

# Define locally
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import only the classes, not the variable
from utils.custom_llms import (_GitHubChatCompletionsLLM, 
                               SentenceTransformerEmbeddings, 
                               _HuggingFaceChatLLM)

# Load environment variables

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
        
        # if os.getenv("GROQ_API_KEY"):
        #     providers["Groq"] = {
        #         "models": [
        #             "meta-llama/llama-4-maverick-17b-128e-instruct"
        #         ]
        #     }
        
        if os.getenv("MISTRAL_API_KEY"):
            providers["Mistral"] = {
                "models": [
                    "magistral-small-2506",
                    "mistral-small-2506",
                    # "mistral-small-2503",
                    "mistral-large-2411",
                    "mistral-medium-2505",
                    "ministral-3b-2410",
                    "ministral-8b-2410",
                    "open-mistral-7b",
                    "open-mistral-nemo",
                    "open-mixtral-8x7b",
    
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
                    "meta/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    "Phi-4", 
                    "OpenAI o3",
                    "openai/gpt-4.1",
                    "openai/gpt-4o",
                    "deepseek/DeepSeek-V3-0324",
                    "Grok-3"
                ]
            }

        # Local HuggingFace models
        providers["HuggingFace"] = {
            "models": [
                "Equall/Saul-7B-Instruct-v1",
                # Add other local models here
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
            # elif provider == "Groq":
            #     api_key = os.getenv("GROQ_API_KEY")
            #     if not api_key:
            #         raise ValueError("GROQ_API_KEY not found in environment variables")
                
            #     return ChatGroq(
            #         model=model
            #     )
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
                    "meta/Llama-4-Maverick-17B-128E-Instruct-FP8": "meta/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    "DeepSeek-V3": "deepseek/DeepSeek-V3-0324",
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
            
            elif provider == "HuggingFace":
                return _HuggingFaceChatLLM(
                    model=model,
                    temperature=0.1,
                    top_p=1.0,
                    max_new_tokens=512,
                )
            
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except Exception as e:
            st.error(f"Error creating LLM with provider '{provider}' and model '{model}': {str(e)}")
            return None
        

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
        
        # Azure AI Inference (GitHub token)
        if os.getenv("GITHUB_TOKEN"):
            providers["Azure AI Inference"] = {
                "models": [
                    "text-embedding-3-large",
                    "text-embedding-3-small",
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

            elif provider == "Azure AI Inference":
                from utils.custom_llms import AzureAIInferenceEmbeddings
                
                token = os.getenv("GITHUB_TOKEN")
                if not token:
                    raise ValueError("GITHUB_TOKEN not found in environment variables")
                
                return AzureAIInferenceEmbeddings(
                    model_name=model,
                    token=token
                )

            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")

        except Exception as e:
            st.error(f"Error creating embedding model {provider}/{model}: {e}")
            return None