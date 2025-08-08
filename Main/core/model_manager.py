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
from dotenv import load_dotenv
load_dotenv()
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# # Fireworks integration with protobuf conflict resolution
# FIREWORKS_AVAILABLE = False
# try:
#     # Set protobuf implementation and suppress warnings
#     os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
#     # Suppress protobuf version warnings
#     import warnings
#     warnings.filterwarnings("ignore", message=".*Protobuf gencode version.*")
#     warnings.filterwarnings("ignore", message=".*Please update the gencode.*")
    
#     # Clear any existing protobuf state that might conflict
#     import sys
#     protobuf_modules = [name for name in sys.modules.keys() if 'protobuf' in name or 'grpc' in name]
#     for module_name in protobuf_modules:
#         if module_name in sys.modules:
#             del sys.modules[module_name]
    
#     from langchain_fireworks import ChatFireworks
#     FIREWORKS_AVAILABLE = True
#     print("✅ Fireworks LangChain integration available")
    
# except ImportError:
#     FIREWORKS_AVAILABLE = False
#     print("⚠️ Fireworks not available - install with: pip install langchain-fireworks")
# except Exception as e:
#     FIREWORKS_AVAILABLE = False
#     print(f"⚠️ Fireworks initialization failed: {e}")

# Optional Cerebras import
try:
    from langchain_cerebras import ChatCerebras
    CEREBRAS_AVAILABLE = True
    print("✅ Cerebras integration available")
except ImportError:
    CEREBRAS_AVAILABLE = False
    print("⚠️ Cerebras not available - install with: pip install langchain-cerebras")
except Exception as e:
    CEREBRAS_AVAILABLE = False
    print(f"⚠️ Cerebras initialization failed: {e}")

from langchain_nomic import NomicEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import JinaEmbeddings

def fix_ssl_certificates():
    """Automatically fix SSL certificate issues - standalone function"""
    try:
        # Use certifi bundle for SSL certificates
        certifi_path = certifi.where()
        os.environ['SSL_CERT_FILE'] = certifi_path
        
        # Test SSL context creation
        ssl.create_default_context()
        return True, f"SSL certificates fixed using certifi: {certifi_path}"
    except Exception as e:
        return False, f"Failed to fix SSL certificates: {e}"

# Optional Ollama import
if not os.getenv("OLLAMA_HOST"):
    os.environ["OLLAMA_HOST"] = "http://localhost:11434"

try:
    from langchain_ollama import ChatOllama
    
    # Quick test to verify Ollama is working
    import requests
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    response = requests.get(f"{ollama_host.rstrip('/')}/api/tags", timeout=5)
    
    if response.status_code == 200:
        models_data = response.json()
        available_models = [model['name'] for model in models_data.get('models', [])]
        
        if available_models:
            # Quick test with first model
            test_model = ChatOllama(model=available_models[0], temperature=0.1)
            test_response = test_model.invoke("Hi")
            
            if test_response and hasattr(test_response, 'content'):
                OLLAMA_AVAILABLE = True
                print("✅ Ollama integration verified and working!")
                print(f"📋 Available models: {', '.join(available_models[:3])}{'...' if len(available_models) > 3 else ''}")
            else:
                raise Exception("Model test failed")
        else:
            raise Exception("No Ollama models installed")
    else:
        raise Exception(f"Ollama server not accessible (status {response.status_code})")
        
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not available - install with: pip install langchain_ollama")
except Exception as e:
    OLLAMA_AVAILABLE = False
    print(f"⚠️ Ollama initialization failed: {e}")
    print("💡 Ensure Ollama server is running: ollama serve")


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
                    "moonshotai/kimi-k2-instruct",
                    "meta/llama-3.3-70b-instruct",
                    "microsoft/phi-4-mini-instruct",
                    "microsoft/phi-3.5-moe-instruct",
                    "qwen/qwen2-7b-instruct"
                ]
            }
        
        if os.getenv("GROQ_API_KEY"):
            providers["Groq"] = {
                "models": [
                    "meta-llama/llama-4-maverick-17b-128e-instruct",
                    "llama-3.3-70b-versatile",
                    "qwen/qwen3-32b",
                    "llama-3.1-8b-instant",
                    "moonshotai/kimi-k2-instruct",
                    "deepseek-r1-distill-llama-70b"
                ]
            }
        
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
                    "gpt-4.1-nano-2025-04-14",
                    "gpt-4.1-mini-2025-04-14",
                    "gpt-4.1-2025-04-14",
                    "gpt-4o-2024-11-20",
                    "o3-mini-2025-01-31"
                ]
            }
        
        if os.getenv("GOOGLE_API_KEY"):
            providers["Google"] = {
                "models": [
                    "gemini-1.5-flash",
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                    "gemini-2.5-flash-lite",
                    "gemini-2.5-flash",
                    "gemini-2.5-pro"
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
        
        # Local Ollama models (no API key required)
        if OLLAMA_AVAILABLE:
            providers["Ollama"] = {
                "models": [
                    "qwen2.5:3b",
                    "gemma3:1b", 
                    "granite3.3:2b",
                    "deepseek-r1:1.5b",
                    "llama3.2:3b",
                    "qwen3:1.7b",
                    "smollm2:1.7b"
                ]
            }
        
        if CEREBRAS_AVAILABLE and os.getenv("CEREBRAS_API_KEY"):
            providers["Cerebras"] = {
                "models": [
                    "llama-3.3-70b",
                    "qwen-3-32b",
                    "llama3.1-8b"
                ]
            }
        
        # # Include Fireworks if available and API key is provided (serverless models only)
        # if FIREWORKS_AVAILABLE and os.getenv("FIREWORKS_API_KEY"):
        #     providers["Fireworks"] = {
        #         "models": [
        #             "accounts/fireworks/models/kimi-k2-instruct",
        #             "accounts/fireworks/models/qwen3-235b-a22b",
        #             "accounts/fireworks/models/qwen3-30b-a3b"
        #         ]
        #     }


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
            elif provider == "Groq":
                # Enhanced Groq LLM debugging (similar to OpenAI method)
                api_key = os.getenv("GROQ_API_KEY")
                
                # Log debug information for Groq LLM creation
                print(f"\n{'='*50}")
                print("Groq LLM Model Debug Information")
                print(f"{'='*50}")
                print(f"Model: {model}")
                print(f"API Key Status: {'PROVIDED' if api_key else 'MISSING'}")
                
                if api_key:
                    print(f"API Key Length: {len(api_key)}")
                    print(f"API Key Prefix: {api_key[:10] + '...' if len(api_key) > 10 else 'TOO_SHORT'}")
                    print(f"API Key Format Valid: {api_key.startswith('gsk_')}")
                
                print(f"{'='*50}\n")
                
                # Validate before creating model
                if not api_key:
                    raise ValueError("GROQ_API_KEY environment variable is not set")
                
                if not api_key.startswith("gsk_"):
                    raise ValueError("GROQ_API_KEY appears to be invalid (should start with 'gsk_')")
                
                if len(api_key) < 20:
                    raise ValueError("GROQ_API_KEY appears to be too short")
                
                # Try to create the Groq LLM with detailed error handling
                try:
                    llm_model = ChatGroq(
                        model=model,
                        api_key=api_key,
                        temperature=DEFAULT_TEMPERATURE,
                        top_p=DEFAULT_TOP_P,
                        max_retries=2,
                        request_timeout=60  # Increased from 30s for Gemini model stability
                    )
                    
                    # Test the LLM model with a simple query
                    print("Testing Groq LLM model with sample query...")
                    test_response = llm_model.invoke("test")
                    print(f"✓ Groq LLM model created successfully!")
                    print(f"✓ Test response received: {len(test_response.content)} characters")
                    
                    return llm_model
                    
                except Exception as create_error:
                    print(f"\n❌ Failed to create Groq LLM model:")
                    print(f"Error Type: {type(create_error).__name__}")
                    print(f"Error Message: {str(create_error)}")
                    print(f"Full Traceback:")
                    traceback.print_exc()
                    
                    # Service availability debugging for 503 errors
                    if "503" in str(create_error) or "Service unavailable" in str(create_error):
                        print(f"\n🚨 Groq service unavailability detected:")
                        print(f"  - Status: Service temporarily unavailable (503)")
                        print(f"  - This is a Groq server-side issue, not a client issue")
                        print(f"  - Recommended actions:")
                        print(f"    1. Check Groq service status: https://groqstatus.com/")
                        print(f"    2. Try again in a few minutes")
                        print(f"    3. Switch to a different model/provider temporarily")
                        print(f"    4. Check Groq Discord/Twitter for service updates")
                        print(f"  - Error details: {str(create_error)}")
                    
                    # Additional Windows-specific debugging for [Errno 22]
                    elif "[Errno 22]" in str(create_error) or "Invalid argument" in str(create_error):
                        print(f"\n🔍 Windows-specific [Errno 22] debugging for Groq LLM:")
                        print(f"  - This error often occurs on Windows due to:")
                        print(f"    1. Proxy/firewall blocking connections to groq.com")
                        print(f"    2. Antivirus software interfering with network requests")
                        print(f"    3. Network adapter issues")
                        print(f"    4. SSL/TLS certificate problems")
                        print(f"    5. Windows socket limitations")
                        print(f"  - Try running as administrator")
                        print(f"  - Check Windows Defender/antivirus settings")
                        print(f"  - Verify network connectivity to groq.com")
                        print(f"  - Consider using a different network")
                        print(f"  - Try disabling VPN if enabled")
                        print(f"  - Check Windows firewall settings")
                        
                        # Attempt automatic SSL certificate fix
                        print(f"\n🔧 Attempting automatic SSL certificate fix...")
                        ssl_fixed, ssl_message = fix_ssl_certificates()
                        print(f"SSL fix result: {ssl_message}")
                        
                        if ssl_fixed:
                            print(f"🔄 Retrying Groq LLM model creation after SSL fix...")
                            try:
                                llm_model = ChatGroq(
                                    model=model,
                                    api_key=api_key,
                                    temperature=DEFAULT_TEMPERATURE,
                                    top_p=DEFAULT_TOP_P,
                                    max_retries=2,
                                    request_timeout=60  # Increased from 30s for Gemini model stability
                                )
                                
                                # Test the fixed LLM model
                                test_response = llm_model.invoke("test")
                                print(f"✅ SSL fix successful! Groq LLM model created and tested successfully")
                                return llm_model
                                
                            except Exception as retry_error:
                                print(f"❌ SSL fix didn't resolve the issue: {retry_error}")
                    
                    raise create_error

            elif provider == "Mistral":
                return ChatMistralAI(
                    model=model,
                    api_key=os.getenv("MISTRAL_API_KEY"),
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P
                )
            elif provider == "OpenAI":
                # Enhanced OpenAI LLM debugging (same as embedding method)
                api_key = os.getenv("OPENAI_API_KEY")
                
                # Run comprehensive debugging
                debug_info = EmbeddingManager.debug_openai_connection(api_key)
                
                # Log debug information for LLM creation
                print(f"\n{'='*50}")
                print("OpenAI LLM Model Debug Information")
                print(f"{'='*50}")
                print(f"Model: {model}")
                print(f"API Key Status: {debug_info['api_key_status']}")
                
                if debug_info['api_key_status'] == 'PROVIDED':
                    print(f"API Key Format Valid: {debug_info['api_key_format_valid']}")
                
                print(f"Network Connectivity: {debug_info['network_checks'].get('dns_resolution', 'Unknown')}")
                print(f"HTTP Connectivity: {debug_info['network_checks'].get('http_connectivity', 'Unknown')}")
                print(f"{'='*50}\n")
                
                # Validate before creating model
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable is not set")
                
                if not api_key.startswith("sk-"):
                    raise ValueError("OPENAI_API_KEY appears to be invalid (should start with 'sk-')")
                
                if len(api_key) < 20:
                    raise ValueError("OPENAI_API_KEY appears to be too short")
                
                # Try to create the LLM with detailed error handling
                try:
                    # o3-mini models don't support temperature/top_p parameters
                    if "o3-mini" in model:
                        llm_model = ChatOpenAI(
                            model=model,
                            api_key=api_key,
                            max_retries=2,
                            request_timeout=60  # Increased from 30s for Gemini model stability
                        )
                    else:
                        llm_model = ChatOpenAI(
                            model=model,
                            api_key=api_key,
                            temperature=DEFAULT_TEMPERATURE,
                            top_p=DEFAULT_TOP_P,
                            max_retries=2,
                            request_timeout=60  # Increased from 30s for Gemini model stability
                        )
                    
                    # Test the LLM model with a simple query
                    print("Testing OpenAI LLM model with sample query...")
                    test_response = llm_model.invoke("test")
                    print(f"✓ OpenAI LLM model created successfully!")
                    print(f"✓ Test response received: {len(test_response.content)} characters")
                    
                    return llm_model
                    
                except Exception as create_error:
                    print(f"\n❌ Failed to create OpenAI LLM model:")
                    print(f"Error Type: {type(create_error).__name__}")
                    print(f"Error Message: {str(create_error)}")
                    print(f"Full Traceback:")
                    traceback.print_exc()
                    
                    # Additional Windows-specific debugging for [Errno 22]
                    if "[Errno 22]" in str(create_error) or "Invalid argument" in str(create_error):
                        print(f"\n🔍 Windows-specific [Errno 22] debugging for LLM:")
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
                        print(f"  - Try disabling VPN if enabled")
                        print(f"  - Check Windows firewall settings")
                        
                        # Attempt automatic SSL certificate fix
                        print(f"\n🔧 Attempting automatic SSL certificate fix...")
                        ssl_fixed, ssl_message = fix_ssl_certificates()
                        print(f"SSL fix result: {ssl_message}")
                        
                        if ssl_fixed:
                            print(f"🔄 Retrying LLM model creation after SSL fix...")
                            try:
                                    # o3-mini models don't support temperature/top_p parameters
                                if "o3-mini" in model:
                                    llm_model = ChatOpenAI(
                                        model=model,
                                        api_key=api_key,
                                        max_retries=2,
                                        request_timeout=60  # Increased from 30s for Gemini model stability
                                    )
                                else:
                                    llm_model = ChatOpenAI(
                                        model=model,
                                        api_key=api_key,
                                        temperature=DEFAULT_TEMPERATURE,
                                        top_p=DEFAULT_TOP_P,
                                        max_retries=2,
                                        request_timeout=60  # Increased from 30s for Gemini model stability
                                    )
                                
                                # Test the fixed LLM model
                                test_response = llm_model.invoke("test")
                                print(f"✅ SSL fix successful! LLM model created and tested successfully")
                                return llm_model
                                
                            except Exception as retry_error:
                                    print(f"❌ SSL fix didn't resolve the issue: {retry_error}")
                    
                    raise create_error

            elif provider == "Google":
                return ChatGoogleGenerativeAI(
                    model=model,
                    api_key=os.getenv("GOOGLE_API_KEY"),
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P,
                    # max_retries=3,  # Add explicit retry logic for server-side errors
                    # timeout=120  # Correct parameter for request timeout
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

            elif provider == "Cerebras":
                if not CEREBRAS_AVAILABLE:
                    raise ValueError(
                        "Cerebras is not available. This could be due to:\n"
                        "1. Package not installed (run 'pip install langchain_cerebras')\n"
                        "2. Dependency conflicts with other packages\n"
                        "3. Import errors\n\n"
                        "Check console output for specific error details."
                    )
                
                # Import ChatCerebras locally to avoid top-level import issues
                from langchain_cerebras import ChatCerebras
                return ChatCerebras(
                    model=model,
                    api_key=os.getenv("CEREBRAS_API_KEY"),
                    temperature=DEFAULT_TEMPERATURE,
                    top_p=DEFAULT_TOP_P
                )
            
            # elif provider == "Fireworks":
            #     if not FIREWORKS_AVAILABLE:
            #         raise ValueError(
            #             "Fireworks is not available. This could be due to:\n"
            #             "1. Package not installed (run 'pip install langchain-fireworks')\n"
            #             "2. Protobuf dependency conflicts\n"
            #             "3. Import errors\n\n"
            #             "Check console output for specific error details."
            #         )
                
            #     # Use standard LangChain Fireworks integration
            #     return ChatFireworks(
            #         model=model,
            #         api_key=os.getenv("FIREWORKS_API_KEY"),
            #         temperature=DEFAULT_TEMPERATURE,
            #         top_p=DEFAULT_TOP_P,
            #         max_tokens=4096  # Increased for comprehensive RAG responses with legal context
            #     )
            

            elif provider == "Ollama":
                if not OLLAMA_AVAILABLE:
                    raise ValueError(
                        "Ollama is not available. This could be due to:\n"
                        "1. Ollama server not running (run 'ollama serve')\n"
                        "2. SSL certificate issues on Windows\n"
                        "3. No Ollama models installed\n"
                        "4. Installation problems\n\n"
                        "Check console output for specific error details."
                    )
                
                model_kwargs = {
                    "temperature": DEFAULT_TEMPERATURE,
                    "top_p": DEFAULT_TOP_P,
                }

                if model == "granite3.3:2b":
                    model_kwargs['thinking'] = True
                    print("💡 Enabled 'thinking' parameter for ollama/granite3.3:2b")

                if model == "qwen3:1.7b":
                    model_kwargs['enable_thinking'] = True
                    print("💡 Enabled 'enable_thinking' parameter for ollama/qwen3:1.7b")
                
                return ChatOllama(
                    model=model,
                    **model_kwargs
                )
            
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except Exception as e:
            error_message = str(e)
            
            # Provide more specific error messages for common issues
            if "503" in error_message or "Service unavailable" in error_message:
                st.error(f"🚨 {provider} service is temporarily unavailable")
                st.warning("💡 **Service Issue:** This is a server-side problem. Try again in a few minutes or switch to a different provider.")
                st.info(f"🔗 **Status Check:** Visit the {provider.lower()} status page for service updates")
            elif "401" in error_message or "authentication" in error_message.lower():
                st.error(f"🔑 Authentication failed for {provider}")
                st.warning("💡 **API Key Issue:** Please check your API key is valid and has sufficient credits.")
            elif "429" in error_message or "rate limit" in error_message.lower():
                st.error(f"⏰ Rate limit exceeded for {provider}")
                st.warning("💡 **Rate Limit:** You've hit the API rate limit. Wait a moment and try again.")
            elif "[Errno 22]" in error_message or "Invalid argument" in error_message:
                st.error(f"🌐 Network connection issue with {provider}")
                st.warning("💡 **Network Issue:** This might be a firewall/antivirus issue. Try running as administrator or check network settings.")
            else:
                st.error(f"❌ Error creating LLM with provider '{provider}' and model '{model}': {error_message}")
            

            # Always show the basic provider/model info for debugging
            st.caption(f"Provider: {provider} | Model: {model}")
            
            return None
        

class EmbeddingManager:
    """Manages different embedding providers and models"""
    
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
                        request_timeout=30  # Extended for large batches (was 30s)
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
                        ssl_fixed, ssl_message = fix_ssl_certificates()
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