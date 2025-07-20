from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from LLM inference."""
    text: str
    confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    model_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """Base class for LLM implementations."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if LLM is available."""
        pass
    
    def format_judge_prompt(self, query: str, context: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format a prompt for MTG judge queries."""
        prompt_parts = [
            "You are an expert Magic: The Gathering judge. Answer the following question accurately using the provided context.",
            "",
            f"Question: {query}",
            ""
        ]
        
        # Add relevant cards
        if context.get("cards"):
            prompt_parts.append("Relevant Cards:")
            for card in context["cards"]:
                card_info = f"- {card['name']}"
                if card.get("oracle_text"):
                    card_info += f": {card['oracle_text']}"
                if card.get("type_line"):
                    card_info += f" ({card['type_line']})"
                prompt_parts.append(card_info)
            prompt_parts.append("")
        
        # Add relevant rules
        if context.get("rules"):
            prompt_parts.append("Relevant Rules:")
            for rule in context["rules"]:
                prompt_parts.append(f"- {rule['id']}: {rule['content']}")
            prompt_parts.append("")
        
        # Add relevant rulings
        if context.get("rulings"):
            prompt_parts.append("Relevant Rulings:")
            for ruling in context["rulings"]:
                ruling_text = f"- {ruling.get('card_name', 'Unknown card')}: {ruling['comment']}"
                prompt_parts.append(ruling_text)
            prompt_parts.append("")
        
        prompt_parts.extend([
            "Instructions:",
            "1. Provide a clear, accurate answer based on the context",
            "2. Reference specific rules or rulings when applicable",
            "3. If the question is unclear, ask for clarification",
            "4. If you're not certain, say so and explain what you know",
            "5. Use proper Magic terminology",
            "",
            "Answer:"
        ])
        
        return "\n".join(prompt_parts)