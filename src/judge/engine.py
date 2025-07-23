import json
from typing import Dict, List, Any, Optional
from loguru import logger
from ..llm.base import BaseLLM, LLMResponse
from ..rag.vector_store import FAISSVectorStore


class MTGJudgeEngine:
    """Simplified MTG Judge engine using RAG and tool calling."""
    
    def __init__(self, llm: BaseLLM, vector_store: FAISSVectorStore):
        self.llm = llm
        self.vector_store = vector_store
        self.tools = vector_store.get_retrieval_tools()
    
    def answer_question(self, question: str, stream: bool = False) -> Dict[str, Any]:
        """Answer a Magic: The Gathering judge question."""
        try:
            # Create system prompt for MTG judging
            system_prompt = self._create_system_prompt()
            
            # Combine system prompt with user question
            full_prompt = f"{system_prompt}\n\nUser Question: {question}"
            
            # Generate response with tools
            response = self.llm.generate_with_tools(full_prompt, self.tools, stream=stream)
            
            # Process tool calls if present
            final_answer = response.text
            context_used = {}
            
            if response.metadata and response.metadata.get("tool_calls"):
                tool_results = self._execute_tool_calls(response.metadata["tool_calls"])
                
                if tool_results:
                    # Re-prompt with tool results
                    context_prompt = self._format_context_prompt(question, tool_results)
                    context_response = self.llm.generate(context_prompt, stream=stream)
                    final_answer = context_response.text
                    context_used = tool_results
            
            return {
                "question": question,
                "answer": final_answer,
                "context_used": context_used,
                "model_name": response.model_name,
                "tokens_used": response.tokens_used
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error processing your question: {str(e)}",
                "error": True,
                "context_used": {}
            }
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for MTG judging."""
        return """You are an expert Magic: The Gathering judge. Your role is to provide accurate rulings based on official rules and card interactions.

When answering questions:
1. You may think through the problem first if needed
2. Use the provided tools to search for relevant rules, cards, and rulings
3. After gathering information, provide a clear final answer
4. Reference specific rule numbers when applicable
5. Use proper Magic terminology

You have access to tools that can search:
- Comprehensive Rules: search_rules
- Cards: search_cards  
- Rulings: search_rulings

Process: Think → Search → Answer clearly and definitively."""
    
    def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Execute tool calls and return results."""
        results = {}
        
        for tool_call in tool_calls:
            try:
                function_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
                if function_name == "search_rules":
                    query = arguments.get("query", "")
                    k = arguments.get("k", 5)
                    results["rules"] = self.vector_store.search_rules(query, k)
                    
                elif function_name == "search_cards":
                    query = arguments.get("query", "")
                    k = arguments.get("k", 5)
                    results["cards"] = self.vector_store.search_cards(query, k)
                    
                elif function_name == "search_rulings":
                    query = arguments.get("query", "")
                    k = arguments.get("k", 5)
                    results["rulings"] = self.vector_store.search_rulings(query, k)
                
            except Exception as e:
                logger.warning(f"Failed to execute tool call {function_name}: {e}")
                continue
        
        return results
    
    def _format_context_prompt(self, question: str, tool_results: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format prompt with context from tool results."""
        prompt_parts = [
            "You are an expert Magic: The Gathering judge. Answer the following question using the provided context.",
            "",
            f"Question: {question}",
            ""
        ]
        
        # Add rules context
        if tool_results.get("rules"):
            prompt_parts.append("Relevant Rules:")
            for rule in tool_results["rules"]:
                prompt_parts.append(f"- Rule {rule.get('rule_id', '')}: {rule.get('content', '')}")
            prompt_parts.append("")
        
        # Add cards context
        if tool_results.get("cards"):
            prompt_parts.append("Relevant Cards:")
            for card in tool_results["cards"]:
                card_text = f"- {card.get('name', 'Unknown Card')}"
                if card.get("oracle_text"):
                    card_text += f": {card['oracle_text']}"
                if card.get("type_line"):
                    card_text += f" ({card['type_line']})"
                prompt_parts.append(card_text)
            prompt_parts.append("")
        
        # Add rulings context
        if tool_results.get("rulings"):
            prompt_parts.append("Relevant Rulings:")
            for ruling in tool_results["rulings"]:
                ruling_text = f"- {ruling.get('card_name', 'Unknown')}: {ruling.get('comment', '')}"
                prompt_parts.append(ruling_text)
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Instructions:",
            "1. Provide a clear, accurate answer based on the context above",
            "2. Reference specific rules or rulings when applicable", 
            "3. If the question is unclear, ask for clarification",
            "4. If you're not certain, say so and explain what you know",
            "5. Use proper Magic terminology",
            "",
            "Answer:"
        ])
        
        return "\n".join(prompt_parts)