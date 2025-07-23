import json
import torch
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger
from .base import BaseLLM, LLMResponse


class QwenLLM(BaseLLM):
    """Qwen 3 8B LLM with tool calling support using transformers."""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-8B",
                 temperature: float = 0.7,
                 max_tokens: int = 4096,
                 device: str = "auto",
                 **kwargs):
        super().__init__(model_name, **kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen model and tokenizer."""
        try:
            logger.info(f"Loading Qwen model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                pad_token="<|endoftext|>"
            )
            
            # Determine device
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda:0"
                else:
                    self.device = "cpu"
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if "cuda" in self.device else torch.float32,
                "attn_implementation": "flash_attention_2" if "cuda" in self.device else "eager",
            }
            
            if "cuda" in self.device:
                model_kwargs["device_map"] = {"": 0}  # Force to GPU 0 only
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if "cpu" in self.device:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            self.tokenizer = None
            self.model = None
            raise
    
    def generate(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None, stream: bool = False, **kwargs) -> LLMResponse:
        """Generate response from prompt with optional tool calling."""
        if not self.model or not self.tokenizer:
            return LLMResponse(
                text="Error: Model not loaded",
                model_name=self.model_name
            )
        
        try:
            # Format the prompt for Qwen
            formatted_prompt = self._format_prompt(prompt, tools)
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate with optional streaming
            if stream:
                from transformers import TextStreamer
                streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
                print("Response: ", end="", flush=True)
            else:
                streamer = None
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    top_p=0.8,
                    top_k=20,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                )
            
            if stream:
                print()  # New line after streaming
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Parse tool calls if tools were provided
            tool_calls = []
            if tools and "function_call:" in response:
                tool_calls = self._parse_tool_calls(response)
            
            return LLMResponse(
                text=response,
                model_name=self.model_name,
                tokens_used=outputs[0].shape[0],
                metadata={
                    "tool_calls": tool_calls,
                    "temperature": self.temperature
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return LLMResponse(
                text=f"Error: {str(e)}",
                model_name=self.model_name
            )
    
    def _format_prompt(self, prompt: str, tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """Format prompt for Qwen with optional tool definitions."""
        formatted = f"<|im_start|>system\nYou are a helpful assistant."
        
        if tools:
            formatted += " You have access to the following tools:\n\n"
            for tool in tools:
                formatted += f"Tool: {tool['function']['name']}\n"
                formatted += f"Description: {tool['function']['description']}\n"
                formatted += f"Parameters: {json.dumps(tool['function']['parameters'], indent=2)}\n\n"
            
            formatted += 'To use a tool, respond with: function_call: {"name": "tool_name", "arguments": {"param": "value"}}\n'
            formatted += "After using tools, provide a clear final answer to the user's question."
        
        formatted += f"<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        return formatted
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from model response."""
        tool_calls = []
        
        if "function_call:" in response:
            try:
                # Find the JSON part after function_call:
                start_idx = response.find("function_call:")
                json_part = response[start_idx + 14:].strip()
                
                # Try to extract JSON
                if json_part.startswith("{"):
                    end_idx = json_part.find("}")
                    if end_idx != -1:
                        json_str = json_part[:end_idx + 1]
                        call_data = json.loads(json_str)
                        
                        tool_calls.append({
                            "type": "function",
                            "function": {
                                "name": call_data.get("name", ""),
                                "arguments": json.dumps(call_data.get("arguments", {}))
                            }
                        })
            except Exception as e:
                logger.warning(f"Failed to parse tool call: {e}")
        
        return tool_calls
    
    def generate_with_tools(self, prompt: str, tools: List[Dict[str, Any]], stream: bool = False) -> LLMResponse:
        """Generate response with tool calling capability."""
        return self.generate(prompt, tools=tools, stream=stream)
    
    def is_available(self) -> bool:
        """Check if Qwen model is loaded and available."""
        return self.model is not None and self.tokenizer is not None