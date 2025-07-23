from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import create_engine, or_, and_, func
from sqlalchemy.orm import sessionmaker
from loguru import logger

from .models import Base, Card, Ruling, Rule, TrainingData


class DatabaseQueries:
    def __init__(self, database_url: str = "sqlite:///mtg_judge.db"):
        self.engine = create_engine(database_url)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def search_cards(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for cards by name or text."""
        cards = self.session.query(Card).filter(
            or_(
                Card.name.ilike(f"%{query}%"),
                Card.oracle_text.ilike(f"%{query}%"),
                Card.type_line.ilike(f"%{query}%")
            )
        ).limit(limit).all()
        
        return [self._card_to_dict(card) for card in cards]
    
    def get_card_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific card by exact name."""
        card = self.session.query(Card).filter(Card.name == name).first()
        return self._card_to_dict(card) if card else None
    
    def get_card_suggestions(self, partial_name: str, limit: int = 10) -> List[str]:
        """Get card name suggestions for typeahead."""
        cards = self.session.query(Card.name).filter(
            Card.name.ilike(f"{partial_name}%")
        ).distinct().limit(limit).all()
        
        return [card.name for card in cards]
    
    def get_rulings_for_card(self, oracle_id: str) -> List[Dict[str, Any]]:
        """Get all rulings for a specific card."""
        rulings = self.session.query(Ruling).filter(
            Ruling.oracle_id == oracle_id
        ).order_by(Ruling.published_at.desc()).all()
        
        return [self._ruling_to_dict(ruling) for ruling in rulings]
    
    def search_rules(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search comprehensive rules."""
        rules = self.session.query(Rule).filter(
            or_(
                Rule.content.ilike(f"%{query}%"),
                Rule.title.ilike(f"%{query}%")
            )
        ).limit(limit).all()
        
        return [self._rule_to_dict(rule) for rule in rules]
    
    def get_rule_by_id(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific rule by ID."""
        rule = self.session.query(Rule).filter(Rule.id == rule_id).first()
        return self._rule_to_dict(rule) if rule else None
    
    def get_subrules(self, rule_id: str) -> List[Dict[str, Any]]:
        """Get all subrules for a given rule ID."""
        subrules = self.session.query(Rule).filter(Rule.parent_rule == rule_id).all()
        return [self._rule_to_dict(rule) for rule in subrules]
    
    def search_rulings(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search rulings text."""
        rulings = self.session.query(Ruling).join(Card).filter(
            or_(
                Ruling.comment.ilike(f"%{query}%"),
                Card.name.ilike(f"%{query}%")
            )
        ).limit(limit).all()
        
        return [self._ruling_to_dict(ruling) for ruling in rulings]
    
    def add_training_data(self, query: str, response: str, is_correct: bool = True, 
                         corrected_response: str = None, feedback: str = None) -> str:
        """Add training data from user interactions."""
        training_id = f"train_{hash(query + response)}"
        
        existing = self.session.query(TrainingData).filter_by(id=training_id).first()
        if existing:
            # Update existing record
            existing.is_correct = is_correct
            if corrected_response:
                existing.corrected_response = corrected_response
            if feedback:
                existing.feedback = feedback
        else:
            # Create new record
            training_data = TrainingData(
                id=training_id,
                query=query,
                response=response,
                is_correct=is_correct,
                corrected_response=corrected_response,
                feedback=feedback
            )
            self.session.add(training_data)
        
        self.session.commit()
        return training_id
    
    def get_training_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get training data for DSPy."""
        training_data = self.session.query(TrainingData).limit(limit).all()
        return [self._training_to_dict(td) for td in training_data]
    
    def get_context_for_query(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get relevant context (cards, rules, rulings) for a query."""
        context = {
            "cards": self.search_cards(query, limit=5),
            "rules": self.search_rules(query, limit=5),
            "rulings": self.search_rulings(query, limit=5)
        }
        return context
    
    def _card_to_dict(self, card: Card) -> Dict[str, Any]:
        """Convert Card model to dictionary."""
        return {
            "id": card.id,
            "oracle_id": card.oracle_id,
            "name": card.name,
            "oracle_text": card.oracle_text,
            "type_line": card.type_line,
            "mana_cost": card.mana_cost,
            "cmc": card.cmc,
            "power": card.power,
            "toughness": card.toughness,
            "loyalty": card.loyalty,
            "defense": card.defense,
            "colors": card.colors,
            "color_identity": card.color_identity,
            "keywords": card.keywords,
            "legalities": card.legalities,
            "set_code": card.set_code,
            "set_name": card.set_name,
            "rarity": card.rarity,
            "artist": card.artist,
            "flavor_text": card.flavor_text
        }
    
    def _ruling_to_dict(self, ruling: Ruling) -> Dict[str, Any]:
        """Convert Ruling model to dictionary."""
        return {
            "id": ruling.id,
            "oracle_id": ruling.oracle_id,
            "source": ruling.source,
            "published_at": ruling.published_at.isoformat(),
            "comment": ruling.comment,
            "card_name": ruling.card.name if ruling.card else None
        }
    
    def _rule_to_dict(self, rule: Rule) -> Dict[str, Any]:
        """Convert Rule model to dictionary."""
        return {
            "id": rule.id,
            "section": rule.section,
            "title": rule.title,
            "content": rule.content,
            "parent_rule": rule.parent_rule
        }
    
    def _training_to_dict(self, training: TrainingData) -> Dict[str, Any]:
        """Convert TrainingData model to dictionary."""
        return {
            "id": training.id,
            "query": training.query,
            "response": training.response,
            "is_correct": training.is_correct,
            "corrected_response": training.corrected_response,
            "created_at": training.created_at.isoformat(),
            "feedback": training.feedback
        }
    
    def get_all_rules(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all rules from the database."""
        query = self.session.query(Rule)
        if limit:
            query = query.limit(limit)
        rules = query.all()
        return [self._rule_to_dict(rule) for rule in rules]
    
    def get_all_cards(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all cards from the database."""
        query = self.session.query(Card)
        if limit:
            query = query.limit(limit)
        cards = query.all()
        return [self._card_to_dict(card) for card in cards]
    
    def get_all_rulings(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all rulings from the database."""
        query = self.session.query(Ruling)
        if limit:
            query = query.limit(limit)
        rulings = query.all()
        return [self._ruling_to_dict(ruling) for ruling in rulings]

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        return {
            "cards": self.session.query(Card).count(),
            "rulings": self.session.query(Ruling).count(),
            "rules": self.session.query(Rule).count(),
            "training_data": self.session.query(TrainingData).count()
        }
    
    def close(self):
        """Close database session."""
        self.session.close()