import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, Dict, Any
from loguru import logger

from .base import BaseLLM, LLMResponse


class LocalLLM(BaseLLM):
    """Local LLM implementation using Transformers."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", **kwargs):
        super().__init__(model_name, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.max_length = kwargs.get("max_length", 2048)
        self.temperature = kwargs.get("temperature", 0.7)
        self.load_model()
    
    def load_model(self):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model {self.model_name} on {self.device}")
            
            # Configure quantization for GPU memory efficiency
            quantization_config = None
            if self.device == "cuda" and self.config.get("use_quantization", True):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left"
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "trust_remote_code": True
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from prompt."""
        if not self.is_available():
            return LLMResponse(
                text="Local LLM is not available. Please check the model configuration.",
                model_name=self.model_name
            )
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", 512),
                "temperature": kwargs.get("temperature", self.temperature),
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                "top_p": kwargs.get("top_p", 0.9),
            }
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(inputs, **generation_kwargs)
            
            # Decode response
            response_tokens = outputs[0][inputs.shape[1]:]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return LLMResponse(
                text=response_text.strip(),
                tokens_used=len(response_tokens),
                model_name=self.model_name,
                metadata={
                    "device": self.device,
                    "input_tokens": inputs.shape[1],
                    "output_tokens": len(response_tokens)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return LLMResponse(
                text=f"Error generating response: {str(e)}",
                model_name=self.model_name
            )
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.model is not None and self.tokenizer is not None
    
    @classmethod
    def get_recommended_models(cls) -> Dict[str, str]:
        """Get recommended models for different use cases."""
        return {
            "small": "microsoft/DialoGPT-small",
            "medium": "microsoft/DialoGPT-medium", 
            "large": "microsoft/DialoGPT-large",
            "code": "codellama/CodeLlama-7b-Instruct-hf",
            "chat": "meta-llama/Llama-2-7b-chat-hf",
            "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
        }


class HuggingFaceLLM(LocalLLM):
    """Enhanced local LLM with better prompt handling for instruction-following models."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1", **kwargs):
        super().__init__(model_name, **kwargs)
    
    def format_instruction_prompt(self, instruction: str, context: str = "") -> str:
        """Format prompt for instruction-following models."""
        if "llama" in self.model_name.lower():
            # Llama format
            if context:
                return f"<s>[INST] {context}\n\n{instruction} [/INST]"
            else:
                return f"<s>[INST] {instruction} [/INST]"
        
        elif "mistral" in self.model_name.lower():
            # Mistral format
            if context:
                return f"<s>[INST] {context}\n\n{instruction} [/INST]"
            else:
                return f"<s>[INST] {instruction} [/INST]"
        
        else:
            # Generic format
            if context:
                return f"Context: {context}\n\nInstruction: {instruction}\n\nResponse:"
            else:
                return f"Instruction: {instruction}\n\nResponse:"
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response with proper formatting."""
        # Check if this looks like a raw instruction
        if not any(marker in prompt.lower() for marker in ["[inst]", "instruction:", "context:"]):
            prompt = self.format_instruction_prompt(prompt)
        
        return super().generate(prompt, **kwargs)