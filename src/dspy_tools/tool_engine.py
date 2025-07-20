"""Tool-based judge engine that actively queries the database."""

import dspy
import json
from typing import Dict, List, Any, Optional
from loguru import logger

from .tool_signatures import (
    QueryCards, QueryRules, QueryRulings, QuerySpecificCard, 
    QuerySpecificRule, PlanQueries, SynthesizeAnswer, ToolBasedJudgeAnswer
)
from ..database.queries import DatabaseQueries


class DatabaseTool:
    """Base class for database query tools."""
    
    def __init__(self, db_queries: DatabaseQueries):
        self.db = db_queries
    
    def format_card_details(self, card: dict) -> str:
        """Format card details in a comprehensive, MTG judge-friendly way."""
        details = []
        
        # Basic info
        details.append(f"**{card['name']}** - {card['mana_cost'] or '(No mana cost)'}")
        details.append(f"Type: {card['type_line']}")
        
        # Stats based on card type
        if card['power'] is not None and card['toughness'] is not None:
            details.append(f"Power/Toughness: {card['power']}/{card['toughness']}")
        
        if card['loyalty'] is not None:
            details.append(f"Starting Loyalty: {card['loyalty']}")
        
        if card['defense'] is not None:
            details.append(f"Defense: {card['defense']}")
        
        # Mana info
        details.append(f"Mana Value: {card['cmc']}")
        
        # Colors
        if card['colors']:
            details.append(f"Colors: {', '.join(card['colors'])}")
        else:
            details.append("Colors: Colorless")
        
        if card['color_identity']:
            details.append(f"Color Identity: {', '.join(card['color_identity'])}")
        
        # Oracle text
        if card['oracle_text']:
            details.append(f"Oracle Text: {card['oracle_text']}")
        
        # Keywords
        if card['keywords']:
            details.append(f"Keywords: {', '.join(card['keywords'])}")
        
        # Set info
        details.append(f"Set: {card['set_name']} ({card['set_code']}) - {card['rarity']}")
        
        # Legalities (key formats only)
        legal_formats = []
        key_formats = ['standard', 'modern', 'legacy', 'vintage', 'commander', 'pioneer']
        for fmt in key_formats:
            if card['legalities'].get(fmt) == 'legal':
                legal_formats.append(fmt.title())
        
        if legal_formats:
            details.append(f"Legal in: {', '.join(legal_formats)}")
        
        return '\n'.join(details)


class CardQueryTool(DatabaseTool):
    """Tool for querying cards."""
    
    def __call__(self, query: str, limit: int = 5, format_type: str = "json") -> str:
        """Query cards and return formatted data."""
        try:
            cards = self.db.search_cards(query, limit=limit)
            
            if format_type == "detailed" and cards:
                # Return human-readable detailed format for first few cards
                formatted_cards = []
                for i, card in enumerate(cards[:3]):  # Limit to first 3 for readability
                    formatted_cards.append(f"Card {i+1}:\n{self.format_card_details(card)}")
                
                if len(cards) > 3:
                    formatted_cards.append(f"\n... and {len(cards) - 3} more cards found.")
                
                return '\n\n'.join(formatted_cards)
            else:
                # Return JSON format
                return json.dumps(cards, indent=2)
        except Exception as e:
            logger.error(f"Error querying cards: {e}")
            return json.dumps({"error": str(e)})


class RuleQueryTool(DatabaseTool):
    """Tool for querying rules."""
    
    def __call__(self, query: str, limit: int = 5) -> str:
        """Query rules and return JSON."""
        try:
            rules = self.db.search_rules(query, limit=limit)
            return json.dumps(rules, indent=2)
        except Exception as e:
            logger.error(f"Error querying rules: {e}")
            return json.dumps({"error": str(e)})


class RulingQueryTool(DatabaseTool):
    """Tool for querying rulings."""
    
    def __call__(self, query: str, limit: int = 5) -> str:
        """Query rulings and return JSON."""
        try:
            rulings = self.db.search_rulings(query, limit=limit)
            return json.dumps(rulings, indent=2)
        except Exception as e:
            logger.error(f"Error querying rulings: {e}")
            return json.dumps({"error": str(e)})


class SpecificCardTool(DatabaseTool):
    """Tool for getting specific card data."""
    
    def __call__(self, card_name: str, include_rulings: bool = False, format_type: str = "detailed") -> str:
        """Get specific card data and return formatted data."""
        try:
            card = self.db.get_card_by_name(card_name)
            if not card:
                return json.dumps({"error": f"Card '{card_name}' not found"})
            
            if include_rulings:
                rulings = self.db.get_rulings_for_card(card["oracle_id"])
                card["rulings"] = rulings
            
            if format_type == "detailed":
                formatted_card = self.format_card_details(card)
                
                # Add rulings if requested
                if include_rulings and card.get("rulings"):
                    formatted_card += "\n\nOfficial Rulings:"
                    for i, ruling in enumerate(card["rulings"][:5], 1):  # Show max 5 rulings
                        formatted_card += f"\n{i}. {ruling['comment']} ({ruling['published_at'][:10]})"
                
                return formatted_card
            else:
                # Return JSON format
                return json.dumps(card, indent=2)
        except Exception as e:
            logger.error(f"Error getting card {card_name}: {e}")
            return json.dumps({"error": str(e)})


class SpecificRuleTool(DatabaseTool):
    """Tool for getting specific rule data."""
    
    def __call__(self, rule_number: str) -> str:
        """Get specific rule data and return JSON."""
        try:
            rule = self.db.get_rule_by_id(rule_number)
            if not rule:
                return json.dumps({"error": f"Rule '{rule_number}' not found"})
            
            # Also get subrules
            subrules = self.db.get_subrules(rule_number)
            if subrules:
                rule["subrules"] = subrules
            
            return json.dumps(rule, indent=2)
        except Exception as e:
            logger.error(f"Error getting rule {rule_number}: {e}")
            return json.dumps({"error": str(e)})


class ToolBasedJudgeEngine(dspy.Module):
    """Judge engine that uses tools to actively query the database."""
    
    def __init__(self, db_queries: DatabaseQueries):
        super().__init__()
        self.db = db_queries
        
        # Initialize tools
        self.card_tool = CardQueryTool(db_queries)
        self.rule_tool = RuleQueryTool(db_queries)
        self.ruling_tool = RulingQueryTool(db_queries)
        self.specific_card_tool = SpecificCardTool(db_queries)
        self.specific_rule_tool = SpecificRuleTool(db_queries)
        
        # DSPy components for planning and synthesis
        self.plan_queries = dspy.Predict(PlanQueries)
        self.synthesize_answer = dspy.Predict(SynthesizeAnswer)
        
        # Alternative: direct tool-based answering
        self.direct_answer = dspy.Predict(ToolBasedJudgeAnswer)
    
    def forward(self, question: str) -> Dict[str, Any]:
        """Process question using tool-based approach."""
        logger.info(f"Processing question with tools: {question[:50]}...")
        
        # Approach 1: Plan-Execute-Synthesize
        query_results = self._plan_and_execute(question)
        
        # Approach 2: Direct tool-aware answering
        # This tells the LLM about available tools and lets it "call" them
        tool_description = """
Available tools:
- QueryCards(query, limit): Search for cards by name/text
- QueryRules(query, limit): Search comprehensive rules  
- QueryRulings(query, limit): Search official rulings
- QuerySpecificCard(card_name, include_rulings): Get specific card data
- QuerySpecificRule(rule_number): Get specific rule data

Use these tools to find accurate information before answering.
"""
        
        try:
            # Convert query results to context format
            context_description = self._format_query_results(query_results)
            
            # Get tool-aware answer with actual results
            direct_result = self.direct_answer(
                question=question,
                available_tools=f"{tool_description}\n\nQuery Results:\n{context_description}"
            )
            
            return {
                "question": question,
                "answer": direct_result.answer,
                "approach": "tool_based",
                "query_results": query_results,
                "tools_available": True
            }
            
        except Exception as e:
            logger.error(f"Error in tool-based processing: {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "error": True
            }
    
    def _plan_and_execute(self, question: str) -> Dict[str, Any]:
        """Plan queries, execute them, and return results."""
        try:
            # Plan what queries are needed
            plan_result = self.plan_queries(question=question)
            
            # Parse the plan
            try:
                query_plan = json.loads(plan_result.query_plan)
            except:
                # Fallback to basic queries if planning fails
                query_plan = self._fallback_plan(question)
            
            # Execute planned queries
            results = {}
            for query_item in query_plan:
                tool_name = query_item.get("tool_name", "")
                query_text = query_item.get("query_text", "")
                
                if tool_name == "QueryCards":
                    results["cards"] = self.card_tool(query_text, limit=5)
                elif tool_name == "QueryRules":
                    results["rules"] = self.rule_tool(query_text, limit=5)
                elif tool_name == "QueryRulings":
                    results["rulings"] = self.ruling_tool(query_text, limit=5)
                elif tool_name == "QuerySpecificCard":
                    results["specific_card"] = self.specific_card_tool(query_text, include_rulings=True)
                elif tool_name == "QuerySpecificRule":
                    results["specific_rule"] = self.specific_rule_tool(query_text)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in plan and execute: {e}")
            return self._fallback_queries(question)
    
    def _fallback_plan(self, question: str) -> List[Dict[str, str]]:
        """Fallback query plan when planning fails."""
        plan = []
        
        # Extract potential card names (capitalized words)
        words = question.split()
        potential_cards = [w for w in words if w[0].isupper() and len(w) > 3]
        
        # Always search for rules with key terms
        plan.append({
            "tool_name": "QueryRules",
            "query_text": question,
            "reason": "Search for relevant rules"
        })
        
        # Search for cards
        plan.append({
            "tool_name": "QueryCards", 
            "query_text": question,
            "reason": "Search for relevant cards"
        })
        
        # Search for rulings
        plan.append({
            "tool_name": "QueryRulings",
            "query_text": question,
            "reason": "Search for relevant rulings"
        })
        
        return plan
    
    def _fallback_queries(self, question: str) -> Dict[str, Any]:
        """Fallback queries when planning completely fails."""
        results = {}
        
        try:
            results["cards"] = self.card_tool(question, limit=5)
        except:
            results["cards"] = "[]"
        
        try:
            results["rules"] = self.rule_tool(question, limit=5)
        except:
            results["rules"] = "[]"
        
        try:
            results["rulings"] = self.ruling_tool(question, limit=5)
        except:
            results["rulings"] = "[]"
        
        return results
    
    def _format_query_results(self, query_results: Dict[str, Any]) -> str:
        """Format query results into readable context for the model."""
        formatted = []
        
        # Format cards
        if 'cards' in query_results:
            try:
                import json
                cards_data = json.loads(query_results['cards']) if isinstance(query_results['cards'], str) else query_results['cards']
                if cards_data and len(cards_data) > 0:
                    formatted.append("=== CARDS ===")
                    for card in cards_data[:3]:  # Limit to first 3 cards
                        if isinstance(card, dict):
                            name = card.get('name', 'Unknown')
                            oracle = card.get('oracle_text', 'No text')
                            formatted.append(f"• {name}: {oracle}")
                        else:
                            formatted.append(f"• {str(card)}")
            except:
                formatted.append("=== CARDS ===\n• Error parsing card data")
        
        # Format rules
        if 'rules' in query_results:
            try:
                import json
                rules_data = json.loads(query_results['rules']) if isinstance(query_results['rules'], str) else query_results['rules']
                if rules_data and len(rules_data) > 0:
                    formatted.append("\n=== RULES ===")
                    for rule in rules_data[:3]:  # Limit to first 3 rules
                        if isinstance(rule, dict):
                            rule_id = rule.get('id', 'Unknown')
                            content = rule.get('content', 'No content')
                            formatted.append(f"• Rule {rule_id}: {content}")
                        else:
                            formatted.append(f"• {str(rule)}")
            except:
                formatted.append("\n=== RULES ===\n• Error parsing rules data")
        
        # Format rulings
        if 'rulings' in query_results:
            try:
                import json
                rulings_data = json.loads(query_results['rulings']) if isinstance(query_results['rulings'], str) else query_results['rulings']
                if rulings_data and len(rulings_data) > 0:
                    formatted.append("\n=== RULINGS ===")
                    for ruling in rulings_data[:3]:  # Limit to first 3 rulings
                        if isinstance(ruling, dict):
                            card_name = ruling.get('card_name', 'Unknown')
                            comment = ruling.get('comment', 'No comment')
                            formatted.append(f"• {card_name}: {comment}")
                        else:
                            formatted.append(f"• {str(ruling)}")
            except:
                formatted.append("\n=== RULINGS ===\n• Error parsing rulings data")
        
        result = '\n'.join(formatted)
        logger.info(f"Formatted context for model: {result[:300]}...")
        return result if result else "No relevant information found in database."


class AdaptiveToolJudgeEngine:
    """Adaptive judge engine using tool-based querying."""
    
    def __init__(self, db_queries: DatabaseQueries, llm):
        self.db = db_queries
        self.llm = llm
        
        # Configure DSPy with the LLM
        self._configure_dspy(llm)
        
        # Initialize tool-based engine
        self.judge_engine = ToolBasedJudgeEngine(db_queries)
    
    def _configure_dspy(self, llm):
        """Configure DSPy with the provided LLM."""
        from ..llm.remote import OllamaLLM, OpenAILLM, AnthropicLLM
        
        if isinstance(llm, OllamaLLM):
            dspy_lm = dspy.LM(
                model=f"ollama_chat/{llm.model_name}",
                api_base=llm.base_url,
                temperature=getattr(llm, 'temperature', 0.7),
                max_tokens=getattr(llm, 'max_tokens', 1500)
            )
        elif isinstance(llm, OpenAILLM):
            dspy_lm = dspy.LM(
                model=f"openai/{llm.model_name}",
                api_key=llm.api_key,
                temperature=getattr(llm, 'temperature', 0.7),
                max_tokens=getattr(llm, 'max_tokens', 1500)
            )
        elif isinstance(llm, AnthropicLLM):
            dspy_lm = dspy.LM(
                model=f"anthropic/{llm.model_name}",
                api_key=llm.api_key,
                temperature=getattr(llm, 'temperature', 0.7),
                max_tokens=getattr(llm, 'max_tokens', 1500)
            )
        else:
            # Fallback wrapper
            class DSPyLLMWrapper(dspy.BaseLM):
                def __init__(self, llm):
                    self.llm = llm
                    super().__init__(
                        model=getattr(llm, 'model_name', 'custom'),
                        model_type='chat'
                    )
                
                def forward(self, prompt=None, messages=None, **kwargs):
                    if messages:
                        prompt = "\n".join([f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in messages])
                    
                    response = self.llm.generate(prompt, **kwargs)
                    
                    return {
                        "choices": [{
                            "message": {
                                "content": response.text,
                                "role": "assistant"
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "total_tokens": getattr(response, 'tokens_used', 0),
                            "prompt_tokens": 0,
                            "completion_tokens": getattr(response, 'tokens_used', 0)
                        }
                    }
            
            dspy_lm = DSPyLLMWrapper(llm)
        
        dspy.settings.configure(lm=dspy_lm)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using tool-based approach."""
        try:
            result = self.judge_engine(question)
            
            # Store interaction for training
            self.db.add_training_data(
                query=question,
                response=result["answer"],
                is_correct=True  # Assume correct until user feedback
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in tool-based answering: {e}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "error": True
            }
    
    def submit_correction(self, question: str, original_answer: str, 
                         corrected_answer: str, feedback: str = None) -> bool:
        """Submit a correction for training."""
        try:
            self.db.add_training_data(
                query=question,
                response=original_answer,
                is_correct=False,
                corrected_response=corrected_answer,
                feedback=feedback
            )
            
            logger.info(f"Correction submitted for question: {question[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting correction: {e}")
            return False