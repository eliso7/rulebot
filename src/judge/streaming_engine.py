import asyncio
import re
from typing import Dict, List, Any, AsyncGenerator
from loguru import logger
from .engine import MTGJudgeEngine


class StreamingMTGJudgeEngine(MTGJudgeEngine):
    """Streaming version of MTG Judge Engine for real-time chat responses."""
    
    async def stream_answer(self, question: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream the answer with thinking and final response separated."""
        try:
            # Create system prompt
            system_prompt = self._create_system_prompt()
            full_prompt = f"{system_prompt}\n\nUser Question: {question}"
            
            # Start streaming the response
            response_buffer = ""
            thinking_buffer = ""
            answer_buffer = ""
            is_in_thinking = False
            thinking_complete = False
            
            # Use a simple approach - capture the full response first, then parse
            # In a real implementation, you'd want to stream token by token
            response = self.llm.generate_with_tools(full_prompt, self.tools, stream=False)
            full_response = response.text
            
            # Parse thinking vs answer sections
            thinking_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
            if thinking_match:
                thinking_text = thinking_match.group(1).strip()
                # Stream thinking content
                for i, char in enumerate(thinking_text):
                    thinking_buffer += char
                    if i % 10 == 0 or i == len(thinking_text) - 1:  # Send chunks
                        yield {
                            "type": "thinking",
                            "content": char if i > 0 else thinking_buffer
                        }
                        await asyncio.sleep(0.01)  # Small delay for streaming effect
                
                # Remove thinking from main response
                answer_text = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()
            else:
                answer_text = full_response
            
            # Process tool calls if present
            if response.metadata and response.metadata.get("tool_calls"):
                tool_results = self._execute_tool_calls(response.metadata["tool_calls"])
                
                if tool_results:
                    # Create context prompt and get final answer
                    context_prompt = self._format_context_prompt(question, tool_results)
                    context_response = self.llm.generate(context_prompt, stream=False)
                    answer_text = context_response.text
            
            # Clean up any remaining function calls from the answer
            answer_text = re.sub(r'function_call:.*?}', '', answer_text, flags=re.DOTALL)
            answer_text = answer_text.strip()
            
            # Stream the final answer
            for i, char in enumerate(answer_text):
                answer_buffer += char
                if i % 5 == 0 or i == len(answer_text) - 1:  # Send smaller chunks for answer
                    yield {
                        "type": "answer",
                        "content": char if i > 0 else answer_buffer
                    }
                    await asyncio.sleep(0.02)  # Slightly slower for reading
            
            # Signal completion
            yield {"type": "done"}
            
        except Exception as e:
            logger.error(f"Error in streaming answer: {e}")
            yield {
                "type": "error",
                "content": f"I encountered an error: {str(e)}"
            }


async def simulate_streaming_response(text: str, chunk_size: int = 5) -> AsyncGenerator[str, None]:
    """Simulate streaming by yielding text in chunks."""
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        yield chunk
        await asyncio.sleep(0.03)  # Simulate network delay