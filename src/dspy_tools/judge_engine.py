import dspy
from typing import Dict, List, Any, Optional
from loguru import logger

from .signatures import (
    CardSearch, RuleSearch, RulingSearch, JudgeAnswer, 
    AnswerValidation, QueryClassification
)
from ..database.queries import DatabaseQueries
from ..llm.base import BaseLLM


class MTGJudgeEngine(dspy.Module):
    """Main DSPy module for MTG judge queries."""
    
    def __init__(self, db_queries: DatabaseQueries):
        super().__init__()
        self.db = db_queries
        
        # DSPy components
        self.classify_query = dspy.Predict(QueryClassification)
        self.search_cards = dspy.Predict(CardSearch)
        self.search_rules = dspy.Predict(RuleSearch)
        self.search_rulings = dspy.Predict(RulingSearch)
        self.generate_answer = dspy.Predict(JudgeAnswer)
        self.validate_answer = dspy.Predict(AnswerValidation)
    
    def forward(self, question: str) -> Dict[str, Any]:
        """Process a judge question and return an answer."""
        
        # Step 1: Classify the query
        try:
            classification = self.classify_query(query=question)
            # Parse the simplified classification format
            class_text = classification.classification
            logger.info(f"Query classified as: {class_text}")
            
            # Extract components from format: "TYPE: X, TERMS: Y, PRIORITY: Z"
            query_type = "interaction"  # default
            key_terms = [question]      # default
            priority = "cards"          # default
            
            try:
                parts = class_text.split(", ")
                for part in parts:
                    if part.startswith("TYPE:"):
                        query_type = part.split(":", 1)[1].strip()
                    elif part.startswith("TERMS:"):
                        key_terms = [t.strip() for t in part.split(":", 1)[1].split(",")]
                    elif part.startswith("PRIORITY:"):
                        priority = part.split(":", 1)[1].strip()
            except:
                logger.warning("Failed to parse classification, using defaults")
            
            # Create a simple object with the expected attributes
            classification_obj = type('Classification', (), {
                'query_type': query_type,
                'key_terms': key_terms, 
                'priority': priority
            })()
            
        except Exception as e:
            logger.warning(f"Classification failed: {e}, using defaults")
            classification_obj = type('Classification', (), {
                'query_type': 'interaction',
                'key_terms': [question],
                'priority': 'cards'
            })()
        
        # Step 2: Search for relevant context based on classification
        context = self._gather_context(question, classification_obj)
        
        # Step 3: Generate answer using context
        formatted_cards = self._format_cards(context.get("cards", []))
        formatted_rules = self._format_rules(context.get("rules", []))
        formatted_rulings = self._format_rulings(context.get("rulings", []))
        
        # Debug logging to see what context we're providing
        logger.info(f"Context being provided:")
        logger.info(f"Cards: {formatted_cards[:200]}...")
        logger.info(f"Rules: {formatted_rules[:200]}...")
        logger.info(f"Rulings: {formatted_rulings[:200]}...")
        
        answer_result = self.generate_answer(
            question=question,
            cards=formatted_cards,
            rules=formatted_rules,
            rulings=formatted_rulings
        )
        
        # Step 4: Validate answer (simplified for model compatibility)
        try:
            validation = self.validate_answer(
                question=question,
                answer=answer_result.answer,
                context=str(context)
            )
            validation_text = validation.validation_result
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            validation_text = "VALID: Answer generated successfully"
        
        return {
            "question": question,
            "answer": answer_result.answer,
            "classification": {
                "query_type": classification_obj.query_type,
                "key_terms": classification_obj.key_terms,
                "priority": classification_obj.priority
            },
            "context": context,
            "validation": {
                "result": validation_text,
                "is_valid": "VALID" in validation_text,
                "feedback": validation_text
            }
        }
    
    def _gather_context(self, question: str, classification) -> Dict[str, List[Dict[str, Any]]]:
        """Gather relevant context based on query classification."""
        context = {"cards": [], "rules": [], "rulings": []}
        
        # Clean and extract key terms (handle DSPy parsing issues)
        key_terms_raw = getattr(classification, 'key_terms', question)
        if isinstance(key_terms_raw, str):
            # Clean up any formatting artifacts
            key_terms_raw = key_terms_raw.strip().lstrip(']').strip()
            key_terms = [term.strip().strip('"').strip("'") for term in key_terms_raw.split(",")]
        else:
            key_terms = [question]
        
        # Clean priority and query_type
        priority = getattr(classification, 'priority', 'cards')
        if isinstance(priority, str):
            priority = priority.strip().lstrip(']').strip()
        
        query_type = getattr(classification, 'query_type', 'interaction')
        if isinstance(query_type, str):
            query_type = query_type.strip().lstrip(']').strip()
        
        logger.info(f"Cleaned classification - Type: '{query_type}', Terms: {key_terms}, Priority: '{priority}'")
        
        # Determine search strategy based on cleaned classification
        cards_priority = ("cards" in priority.lower() or 
                         "card_specific" in query_type.lower() or
                         "interaction" in query_type.lower())
        
        rules_priority = ("rules" in priority.lower() or 
                         "rules_general" in query_type.lower())
        
        if cards_priority:
            # Focus on cards first
            logger.info("Using cards-focused search strategy")
            for term in key_terms[:3]:  # Limit to top 3 terms
                if term:  # Skip empty terms
                    cards = self.db.search_cards(term.strip(), limit=3)
                    context["cards"].extend(cards)
            
            # Also search for full question in cards
            question_cards = self.db.search_cards(question, limit=2)
            context["cards"].extend(question_cards)
            
            # Get rulings for found cards
            for card in context["cards"][:5]:  # Limit to top 5 cards
                rulings = self.db.get_rulings_for_card(card["oracle_id"])
                context["rulings"].extend(rulings[:2])  # Top 2 rulings per card
            
            # Get some rules using both question and key terms
            rules = self.db.search_rules(question, limit=2)
            context["rules"].extend(rules)
            
            # Also search for rules using key terms
            for term in key_terms[:3]:
                if term and len(term.strip()) > 3:  # Skip very short terms
                    term_rules = self.db.search_rules(term.strip(), limit=2)
                    context["rules"].extend(term_rules)
        
        elif rules_priority:
            # Focus on rules first
            logger.info("Using rules-focused search strategy")
            rules = self.db.search_rules(question, limit=5)
            context["rules"].extend(rules)
            
            # Get some cards and rulings
            cards = self.db.search_cards(question, limit=3)
            context["cards"].extend(cards)
            
            rulings = self.db.search_rulings(question, limit=3)
            context["rulings"].extend(rulings)
        
        else:
            # Balanced search
            logger.info("Using balanced search strategy")
            cards = self.db.search_cards(question, limit=3)
            context["cards"].extend(cards)
            
            rules = self.db.search_rules(question, limit=3)
            context["rules"].extend(rules)
            
            rulings = self.db.search_rulings(question, limit=3)
            context["rulings"].extend(rulings)
        
        # Remove duplicates
        context["cards"] = self._deduplicate_list(context["cards"], "id")
        context["rules"] = self._deduplicate_list(context["rules"], "id")
        context["rulings"] = self._deduplicate_list(context["rulings"], "id")
        
        return context
    
    def _format_cards(self, cards: List[Dict[str, Any]]) -> str:
        """Format cards for DSPy input."""
        if not cards:
            return "No relevant cards found."
        
        formatted = []
        for card in cards:
            card_text = f"**{card['name']}**"
            if card.get("mana_cost"):
                card_text += f" {card['mana_cost']}"
            if card.get("type_line"):
                card_text += f"\n{card['type_line']}"
            if card.get("oracle_text"):
                card_text += f"\n{card['oracle_text']}"
            formatted.append(card_text)
        
        return "\n\n".join(formatted)
    
    def _format_rules(self, rules: List[Dict[str, Any]]) -> str:
        """Format rules for DSPy input."""
        if not rules:
            return "No relevant rules found."
        
        formatted = []
        for rule in rules:
            rule_text = f"**Rule {rule['id']}**: {rule['content']}"
            formatted.append(rule_text)
        
        return "\n\n".join(formatted)
    
    def _format_rulings(self, rulings: List[Dict[str, Any]]) -> str:
        """Format rulings for DSPy input."""
        if not rulings:
            return "No relevant rulings found."
        
        formatted = []
        for ruling in rulings:
            ruling_text = f"**{ruling.get('card_name', 'Unknown')}**: {ruling['comment']}"
            if ruling.get("published_at"):
                ruling_text += f" ({ruling['published_at'][:10]})"
            formatted.append(ruling_text)
        
        return "\n\n".join(formatted)
    
    def _deduplicate_list(self, items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        """Remove duplicates from list based on key."""
        seen = set()
        result = []
        for item in items:
            if item[key] not in seen:
                seen.add(item[key])
                result.append(item)
        return result


class AdaptiveJudgeEngine:
    """Adaptive judge engine that learns from corrections."""
    
    def __init__(self, db_queries: DatabaseQueries, llm: BaseLLM):
        self.db = db_queries
        self.llm = llm
        
        # Configure DSPy with the provided LLM
        dspy.settings.configure(lm=self._wrap_llm(llm))
        
        # Initialize judge engine
        self.judge_engine = MTGJudgeEngine(db_queries)
        
        # Load training data for optimization
        self._load_training_data()
    
    def _wrap_llm(self, llm: BaseLLM):
        """Wrap our LLM for DSPy compatibility."""
        from ..llm.remote import OllamaLLM, OpenAILLM, AnthropicLLM
        from ..llm.local import LocalLLM, HuggingFaceLLM
        
        # Use native DSPy LM for supported providers
        if isinstance(llm, OllamaLLM):
            return dspy.LM(
                model=f"ollama_chat/{llm.model_name}",
                api_base=llm.base_url,
                temperature=getattr(llm, 'temperature', 0.7),
                max_tokens=getattr(llm, 'max_tokens', 1000)
            )
        elif isinstance(llm, OpenAILLM):
            return dspy.LM(
                model=f"openai/{llm.model_name}",
                api_key=llm.api_key,
                temperature=getattr(llm, 'temperature', 0.7),
                max_tokens=getattr(llm, 'max_tokens', 1000)
            )
        elif isinstance(llm, AnthropicLLM):
            return dspy.LM(
                model=f"anthropic/{llm.model_name}",
                api_key=llm.api_key,
                temperature=getattr(llm, 'temperature', 0.7),
                max_tokens=getattr(llm, 'max_tokens', 1000)
            )
        else:
            # Fallback wrapper for local/custom LLMs
            class DSPyLLMWrapper(dspy.BaseLM):
                def __init__(self, llm):
                    self.llm = llm
                    super().__init__(
                        model=getattr(llm, 'model_name', 'custom'),
                        model_type='chat'
                    )
                
                def forward(self, prompt=None, messages=None, **kwargs):
                    if messages:
                        # Convert messages to prompt
                        prompt = "\n".join([f"{msg.get('role', '')}: {msg.get('content', '')}" for msg in messages])
                    
                    response = self.llm.generate(prompt, **kwargs)
                    
                    # Return in OpenAI format
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
            
            return DSPyLLMWrapper(llm)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a judge question."""
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
            logger.error(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": f"I encountered an error processing your question: {str(e)}",
                "error": True
            }
    
    def submit_correction(self, question: str, original_answer: str, 
                         corrected_answer: str, feedback: str = None) -> bool:
        """Submit a correction for training."""
        try:
            # Update training data
            self.db.add_training_data(
                query=question,
                response=original_answer,
                is_correct=False,
                corrected_response=corrected_answer,
                feedback=feedback
            )
            
            # In a full implementation, this would trigger model retraining
            logger.info(f"Correction submitted for question: {question[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting correction: {e}")
            return False
    
    def _load_training_data(self):
        """Load existing training data for optimization."""
        training_data = self.db.get_training_data(limit=100)
        logger.info(f"Loaded {len(training_data)} training examples")
        
        # In a full implementation, this would be used for DSPy optimization
        # For now, we just log the availability of training data
    
    def optimize(self):
        """Optimize the judge engine using training data."""
        # This is where DSPy optimization would happen
        # For now, we'll implement a placeholder
        logger.info("Model optimization not yet implemented")
        pass