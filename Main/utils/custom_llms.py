"""
Custom LLM Implementations
Houses custom LLM classes and wrappers
"""
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional, Union, Callable, Type
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain.schema import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

try:
    from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient
    from azure.ai.inference.models import (SystemMessage as AzureSystem,
                                       UserMessage  as AzureUser,
                                       AssistantMessage as AzureAssistant)
    from azure.core.credentials import AzureKeyCredential
    AZURE_INFERENCE_AVAILABLE = True
except ImportError:
    AZURE_INFERENCE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

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
    _client: Optional["ChatCompletionsClient"] = None

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
    
class _HuggingFaceChatLLM(BaseChatModel):
    """Custom LLM for Hugging Face models"""

    model: str = Field(description="The model name to use")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    top_p: float = Field(default=1.0, description="Top-p for generation")
    max_new_tokens: Optional[int] = Field(default=512, description="Maximum new tokens to generate")

    _pipe: Optional[Any] = None

    def __init__(self, **data):
        super().__init__(**data)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for Hugging Face models. pip install transformers torch")
        
        self._pipe = pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._pipe is None:
            raise RuntimeError("Pipeline not initialized")

        # Correctly format messages for Hugging Face chat templates.
        # Many templates are strict about alternating user/assistant roles
        # and may not support a 'system' role directly.
        system_contents = [m.content for m in messages if m.type == "system"]
        chat_messages = [m for m in messages if m.type != "system"]

        hf_messages = []
        is_first_human = True
        for m in chat_messages:
            if m.type == "human":
                role = "user"
                content = m.content
                if is_first_human and system_contents:
                    content = "\\n".join(system_contents) + "\\n\\n" + content
                    is_first_human = False
                hf_messages.append({"role": role, "content": content})
            elif m.type == "ai":
                role = "assistant"
                hf_messages.append({"role": role, "content": m.content})
        
        if not hf_messages and system_contents:
            hf_messages.append({"role": "user", "content": "\\n".join(system_contents)})

        prompt = self._pipe.tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self._pipe(prompt, do_sample=False)
        content = outputs[0]["generated_text"]
        
        # The output often includes the prompt, so we need to remove it.
        # The actual generated text starts after the last message's content.
        last_message_content = messages[-1].content
        if last_message_content in content:
             # Find the position of last_message_content and get the text after it.
            generated_text_start = content.find(last_message_content) + len(last_message_content)
            # A more robust way to find the start of the assistant's response.
            # The template usually adds something like '<|assistant|>'
            # Looking for what the model adds before its response.
            # Based on HuggingFace template, it adds `add_generation_prompt=True`
            # which for many models adds a marker for the assistant's turn.
            # A simple way is to find the end of the prompt in the output.
            if prompt in content:
                content = content[len(prompt):]
            else:
                 # Fallback if prompt is not exactly in the output due to tokenization differences
                cleaned_output = content.split(last_message_content)[-1]
                content = cleaned_output.strip()

        ai_msg = AIMessage(content=content.strip())
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
        return "huggingface-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
        }

    def __repr__(self):
        return f"<_HuggingFaceChatLLM model={self.model!r}>"

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


class AzureAIInferenceEmbeddings:
    """Custom wrapper for Azure AI Inference embeddings using GitHub token"""
    
    def __init__(self, model_name: str, token: str = None, endpoint: str = "https://models.inference.ai.azure.com"):
        if not AZURE_INFERENCE_AVAILABLE:
            raise ImportError("azure-ai-inference is required for Azure AI embeddings. pip install azure-ai-inference")
        
        # Use provided token or get from environment
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GitHub token is required. Provide token parameter or set GITHUB_TOKEN environment variable")
        
        self.model_name = model_name
        self.endpoint = endpoint
        
        # Initialize the client
        self.client = EmbeddingsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.token)
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            response = self.client.embed(
                input=texts,
                model=self.model_name
            )
            
            # Extract embeddings in the correct order
            embeddings = []
            # Sort by index to ensure correct order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            for item in sorted_data:
                embeddings.append(item.embedding)
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"Azure AI Inference embedding failed: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            response = self.client.embed(
                input=[text],
                model=self.model_name
            )
            
            # Return the first (and only) embedding
            return response.data[0].embedding
            
        except Exception as e:
            raise RuntimeError(f"Azure AI Inference embedding failed: {str(e)}")
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents (currently just a wrapper)"""
        return self.embed_documents(texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query (currently just a wrapper)"""
        return self.embed_query(text)
    
    def __repr__(self):
        return f"<AzureAIInferenceEmbeddings model={self.model_name!r} endpoint={self.endpoint!r}>"

