import httpx
import json
import os
from typing import Optional, Dict, Any
from loguru import logger

from .base import BaseLLM, LLMResponse


class OpenAILLM(BaseLLM):
    """OpenAI API LLM implementation."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = kwargs.get("base_url", "https://api.openai.com/v1")
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 1000)
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not self.api_key:
            return LLMResponse(
                text="OpenAI API key not configured.",
                model_name=self.model_name
            )
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Format for chat models
            if "gpt" in self.model_name:
                data = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens)
                }
                endpoint = "/chat/completions"
            else:
                # Legacy completion models
                data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens)
                }
                endpoint = "/completions"
            
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    json=data,
                    timeout=30.0
                )
                response.raise_for_status()
            
            result = response.json()
            
            if "gpt" in self.model_name:
                text = result["choices"][0]["message"]["content"]
                tokens_used = result.get("usage", {}).get("total_tokens")
            else:
                text = result["choices"][0]["text"]
                tokens_used = result.get("usage", {}).get("total_tokens")
            
            return LLMResponse(
                text=text.strip(),
                tokens_used=tokens_used,
                model_name=self.model_name,
                metadata=result.get("usage", {})
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return LLMResponse(
                text=f"Error calling OpenAI API: {str(e)}",
                model_name=self.model_name
            )
    
    def is_available(self) -> bool:
        """Check if OpenAI API is available."""
        return bool(self.api_key)


class AnthropicLLM(BaseLLM):
    """Anthropic Claude API LLM implementation."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = kwargs.get("base_url", "https://api.anthropic.com/v1")
        self.max_tokens = kwargs.get("max_tokens", 1000)
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic API."""
        if not self.api_key:
            return LLMResponse(
                text="Anthropic API key not configured.",
                model_name=self.model_name
            )
        
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": kwargs.get("max_tokens", self.max_tokens)
            }
            
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=data,
                    timeout=30.0
                )
                response.raise_for_status()
            
            result = response.json()
            text = result["content"][0]["text"]
            tokens_used = result.get("usage", {}).get("input_tokens", 0) + result.get("usage", {}).get("output_tokens", 0)
            
            return LLMResponse(
                text=text.strip(),
                tokens_used=tokens_used,
                model_name=self.model_name,
                metadata=result.get("usage", {})
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return LLMResponse(
                text=f"Error calling Anthropic API: {str(e)}",
                model_name=self.model_name
            )
    
    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        return bool(self.api_key)


class OllamaLLM(BaseLLM):
    """Ollama local API LLM implementation."""
    
    def __init__(self, model_name: str = "llama2", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self.temperature = kwargs.get("temperature", 0.7)
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Ollama API."""
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", 1000)
                }
            }
            
            with httpx.Client() as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json=data,
                    timeout=60.0
                )
                response.raise_for_status()
            
            result = response.json()
            text = result.get("response", "")
            
            return LLMResponse(
                text=text.strip(),
                model_name=self.model_name,
                metadata={
                    "eval_count": result.get("eval_count"),
                    "eval_duration": result.get("eval_duration")
                }
            )
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return LLMResponse(
                text=f"Error calling Ollama API: {str(e)}",
                model_name=self.model_name
            )
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            with httpx.Client() as client:
                response = client.get(f"{self.base_url}/api/tags", timeout=5.0)
                return response.status_code == 200
        except:
            return False


def create_llm(llm_type: str, **kwargs) -> BaseLLM:
    """Factory function to create LLM instances."""
    if llm_type == "local":
        from .local import HuggingFaceLLM
        return HuggingFaceLLM(**kwargs)
    elif llm_type == "openai":
        return OpenAILLM(**kwargs)
    elif llm_type == "anthropic":
        return AnthropicLLM(**kwargs)
    elif llm_type == "ollama":
        return OllamaLLM(**kwargs)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")