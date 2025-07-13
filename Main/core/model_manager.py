"""
Model and Embedding Management
Handles LLM and embedding model instantiation
"""
import os
import sys
import traceback
import requests
import socket
import ssl
from datetime import datetime
import torch
from typing import Dict, List, Optional
import streamlit as st
import certifi
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
from Main.utils.custom_llms import (_GitHubChatCompletionsLLM, 
                                    SentenceTransformerEmbeddings, 
                                    _HuggingFaceChatLLM)
from Main.config.settings import DEFAULT_TEMPERATURE, DEFAULT_TOP_P

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
                    "deepseek-ai/deepseek-r1-0528",
                    "deepseek-ai/deepseek-r1",
                    "meta/llama-3.3-70b-instruct",
                    "microsoft/phi-4-mini-instruct",
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
                    "gpt-4o-mini-2024-07-18",
                    "gpt-4.1-nano-2025-04-14"
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
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P
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
                    api_key=os.getenv("MISTRAL_API_KEY"),
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P
                )
            elif provider == "OpenAI":
                return ChatOpenAI(
                    model=model,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P
                )

            elif provider == "Google":
                return ChatGoogleGenerativeAI(
                    model=model,
                    api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P
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
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P,
                    max_tokens=None,
                )
            
            elif provider == "HuggingFace":
                return _HuggingFaceChatLLM(
                    model=model,
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P,
                    max_new_tokens=1024,
                )
            
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except Exception as e:
            st.error(f"Error creating LLM with provider '{provider}' and model '{model}': {str(e)}")
            return None
        

class EmbeddingManager:
    """Manages different embedding providers and models"""
    
    @staticmethod
    def fix_ssl_certificates():
        """Automatically fix SSL certificate issues"""
        try:
            # Use certifi bundle for SSL certificates
            certifi_path = certifi.where()
            os.environ['SSL_CERT_FILE'] = certifi_path
            
            # Test SSL context creation
            ssl.create_default_context()
            return True, f"SSL certificates fixed using certifi: {certifi_path}"
        except Exception as e:
            return False, f"Failed to fix SSL certificates: {e}"
    
    @staticmethod
    def debug_openai_connection(api_key: str = None):
        """Comprehensive OpenAI connection debugging"""
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "platform": sys.platform,
            "python_version": sys.version,
            "api_key_status": "NOT_PROVIDED",
            "network_checks": {},
            "environment_checks": {},
            "openai_api_test": {}
        }
        
        # Check API key
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            debug_info["api_key_status"] = "PROVIDED"
            debug_info["api_key_length"] = len(api_key)
            debug_info["api_key_prefix"] = api_key[:10] + "..." if len(api_key) > 10 else "TOO_SHORT"
            debug_info["api_key_format_valid"] = api_key.startswith("sk-")
        else:
            debug_info["api_key_status"] = "MISSING"
        
        # Environment checks
        debug_info["environment_checks"] = {
            "PATH": os.environ.get("PATH", "NOT_SET")[:100] + "...",
            "PYTHONPATH": os.environ.get("PYTHONPATH", "NOT_SET"),
            "HTTP_PROXY": os.environ.get("HTTP_PROXY", "NOT_SET"),
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", "NOT_SET"),
            "NO_PROXY": os.environ.get("NO_PROXY", "NOT_SET"),
            "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", "NOT_SET"),
            "OPENAI_API_TYPE": os.environ.get("OPENAI_API_TYPE", "NOT_SET"),
        }
        
        # Network connectivity checks
        try:
            # DNS resolution
            socket.gethostbyname("api.openai.com")
            debug_info["network_checks"]["dns_resolution"] = "SUCCESS"
        except socket.gaierror as e:
            debug_info["network_checks"]["dns_resolution"] = f"FAILED: {str(e)}"
        
        try:
            # Basic HTTP connectivity
            response = requests.get("https://api.openai.com/v1/models", 
                                  headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                                  timeout=10)
            debug_info["network_checks"]["http_connectivity"] = "SUCCESS"
            debug_info["openai_api_test"]["status_code"] = response.status_code
            debug_info["openai_api_test"]["headers"] = dict(response.headers)
            
            if response.status_code == 401:
                debug_info["openai_api_test"]["error"] = "AUTHENTICATION_FAILED"
            elif response.status_code == 200:
                debug_info["openai_api_test"]["error"] = "SUCCESS"
            else:
                debug_info["openai_api_test"]["error"] = f"HTTP_{response.status_code}"
                
        except requests.exceptions.SSLError as e:
            debug_info["network_checks"]["http_connectivity"] = f"SSL_ERROR: {str(e)}"
        except requests.exceptions.ConnectionError as e:
            debug_info["network_checks"]["http_connectivity"] = f"CONNECTION_ERROR: {str(e)}"
        except requests.exceptions.Timeout as e:
            debug_info["network_checks"]["http_connectivity"] = f"TIMEOUT: {str(e)}"
        except Exception as e:
            debug_info["network_checks"]["http_connectivity"] = f"UNKNOWN_ERROR: {str(e)}"
        
        return debug_info
    
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
                    "jina-embeddings-v4",
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
                # Enhanced OpenAI debugging
                api_key = os.getenv("OPENAI_API_KEY")
                
                # Run comprehensive debugging
                debug_info = EmbeddingManager.debug_openai_connection(api_key)
                
                # Log debug information
                print(f"\n{'='*50}")
                print("OpenAI Embedding Model Debug Information")
                print(f"{'='*50}")
                print(f"Timestamp: {debug_info['timestamp']}")
                print(f"Platform: {debug_info['platform']}")
                print(f"Python Version: {debug_info['python_version']}")
                print(f"API Key Status: {debug_info['api_key_status']}")
                
                if debug_info['api_key_status'] == 'PROVIDED':
                    print(f"API Key Length: {debug_info['api_key_length']}")
                    print(f"API Key Prefix: {debug_info['api_key_prefix']}")
                    print(f"API Key Format Valid: {debug_info['api_key_format_valid']}")
                
                print(f"\nEnvironment Variables:")
                for key, value in debug_info['environment_checks'].items():
                    print(f"  {key}: {value}")
                
                print(f"\nNetwork Checks:")
                for key, value in debug_info['network_checks'].items():
                    print(f"  {key}: {value}")
                
                print(f"\nOpenAI API Test:")
                for key, value in debug_info['openai_api_test'].items():
                    if key != 'headers':  # Skip headers to avoid clutter
                        print(f"  {key}: {value}")
                
                print(f"{'='*50}\n")
                
                # Validate before creating model
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is not set")
                
                if not api_key.startswith("sk-"):
                    raise ValueError("OPENAI_API_KEY appears to be invalid (should start with 'sk-')")
                
                if len(api_key) < 20:
                    raise ValueError("OPENAI_API_KEY appears to be too short")
                
                # Try to create the embedding model with detailed error handling
                try:
                    embedding_model = OpenAIEmbeddings(
                        model=model,
                        api_key=api_key,
                        max_retries=2,
                        request_timeout=30
                    )
                    
                    # Test the embedding model with a simple query
                    print("Testing OpenAI embedding model with sample text...")
                    test_embedding = embedding_model.embed_query("test")
                    print(f"✓ OpenAI embedding model created successfully!")
                    print(f"✓ Test embedding dimension: {len(test_embedding)}")
                    
                    return embedding_model
                    
                except Exception as create_error:
                    print(f"\n❌ Failed to create OpenAI embedding model:")
                    print(f"Error Type: {type(create_error).__name__}")
                    print(f"Error Message: {str(create_error)}")
                    print(f"Full Traceback:")
                    traceback.print_exc()
                    
                    # Additional Windows-specific debugging for [Errno 22]
                    if "[Errno 22]" in str(create_error) or "Invalid argument" in str(create_error):
                        print(f"\n🔍 Windows-specific [Errno 22] debugging:")
                        print(f"  - This error often occurs on Windows due to:")
                        print(f"    1. Proxy/firewall blocking connections")
                        print(f"    2. Antivirus software interfering")
                        print(f"    3. Network adapter issues")
                        print(f"    4. SSL/TLS certificate problems")
                        print(f"    5. Windows socket limitations")
                        print(f"  - Try running as administrator")
                        print(f"  - Check Windows Defender/antivirus settings")
                        print(f"  - Verify network connectivity")
                        print(f"  - Consider using a different network")
                        
                        # Attempt automatic SSL certificate fix
                        print(f"\n🔧 Attempting automatic SSL certificate fix...")
                        ssl_fixed, ssl_message = EmbeddingManager.fix_ssl_certificates()
                        print(f"SSL fix result: {ssl_message}")
                        
                        if ssl_fixed:
                            print(f"🔄 Retrying embedding model creation after SSL fix...")
                            try:
                                embedding_model = OpenAIEmbeddings(
                                    model=model,
                                    api_key=api_key,
                                    max_retries=2,
                                    request_timeout=30
                                )
                                
                                # Test the fixed embedding model
                                test_embedding = embedding_model.embed_query("test")
                                print(f"✅ SSL fix successful! Embedding model created with dimension: {len(test_embedding)}")
                                return embedding_model
                                
                            except Exception as retry_error:
                                print(f"❌ SSL fix didn't resolve the issue: {retry_error}")
                    
                    raise create_error
                
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
                from Main.utils.custom_llms import AzureAIInferenceEmbeddings
                
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
            error_msg = (
                f"❌ Error creating embedding model {provider}/{model}:\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Message: {str(e)}\n"
                f"Full Traceback:\n{traceback.format_exc()}"
            )
            print(error_msg)
            st.error(error_msg)
            return None