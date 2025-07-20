import json
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from loguru import logger

from .models import Base, Card, Ruling, Rule, TrainingData


class DataIngestion:
    def __init__(self, database_url: str = "sqlite:///mtg_judge.db"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def load_cards(self, cards_file: Path) -> int:
        """Load cards from all_cards.json file."""
        logger.info(f"Loading cards from {cards_file}")
        
        with open(cards_file, 'r', encoding='utf-8') as f:
            cards_data = json.load(f)
        
        loaded_count = 0
        batch_size = 1000
        
        for i, card_data in enumerate(cards_data):
            try:
                # Check if card already exists
                existing = self.session.query(Card).filter_by(id=card_data['id']).first()
                if existing:
                    continue
                
                # Handle oracle_id - some cards have it in card_faces
                oracle_id = card_data.get('oracle_id')
                if not oracle_id and card_data.get('card_faces'):
                    oracle_id = card_data['card_faces'][0].get('oracle_id')
                
                # Skip cards without oracle_id
                if not oracle_id:
                    logger.warning(f"Skipping card {card_data.get('name', 'unknown')} - no oracle_id found")
                    continue
                
                # Handle double-faced cards
                name = card_data['name']
                oracle_text = card_data.get('oracle_text')
                type_line = card_data.get('type_line')
                mana_cost = card_data.get('mana_cost')
                cmc = card_data.get('cmc')
                power = card_data.get('power')
                toughness = card_data.get('toughness')
                colors = card_data.get('colors', [])
                
                # If it's a double-faced card, use the first face's data
                if card_data.get('card_faces') and not oracle_text:
                    first_face = card_data['card_faces'][0]
                    oracle_text = first_face.get('oracle_text')
                    type_line = first_face.get('type_line')
                    mana_cost = first_face.get('mana_cost')
                    cmc = first_face.get('cmc')
                    power = first_face.get('power')
                    toughness = first_face.get('toughness')
                    colors = first_face.get('colors', [])
                
                card = Card(
                    id=card_data['id'],
                    oracle_id=oracle_id,
                    name=name,
                    oracle_text=oracle_text,
                    type_line=type_line,
                    mana_cost=mana_cost,
                    cmc=cmc,
                    power=power,
                    toughness=toughness,
                    colors=colors,
                    color_identity=card_data.get('color_identity', []),
                    keywords=card_data.get('keywords', []),
                    legalities=card_data.get('legalities', {}),
                    set_code=card_data.get('set'),
                    set_name=card_data.get('set_name'),
                    rarity=card_data.get('rarity'),
                    artist=card_data.get('artist'),
                    flavor_text=card_data.get('flavor_text')
                )
                
                self.session.add(card)
                loaded_count += 1
                
                if loaded_count % batch_size == 0:
                    self.session.commit()
                    logger.info(f"Loaded {loaded_count} cards...")
                    
            except Exception as e:
                logger.error(f"Error loading card {card_data.get('name', 'unknown')}: {e}")
                continue
        
        self.session.commit()
        logger.info(f"Successfully loaded {loaded_count} cards")
        return loaded_count
    
    def load_rulings(self, rulings_file: Path) -> int:
        """Load rulings from rulings.json file."""
        logger.info(f"Loading rulings from {rulings_file}")
        
        with open(rulings_file, 'r', encoding='utf-8') as f:
            rulings_data = json.load(f)
        
        loaded_count = 0
        batch_size = 1000
        
        for i, ruling_data in enumerate(rulings_data):
            try:
                # Create unique ID from oracle_id + published_at + first part of comment
                ruling_id = f"{ruling_data['oracle_id']}_{ruling_data['published_at']}_{hash(ruling_data['comment'][:50])}"
                
                # Check if ruling already exists
                existing = self.session.query(Ruling).filter_by(id=ruling_id).first()
                if existing:
                    continue
                
                ruling = Ruling(
                    id=ruling_id,
                    oracle_id=ruling_data['oracle_id'],
                    source=ruling_data['source'],
                    published_at=datetime.fromisoformat(ruling_data['published_at']),
                    comment=ruling_data['comment']
                )
                
                self.session.add(ruling)
                loaded_count += 1
                
                if loaded_count % batch_size == 0:
                    self.session.commit()
                    logger.info(f"Loaded {loaded_count} rulings...")
                    
            except Exception as e:
                logger.error(f"Error loading ruling: {e}")
                continue
        
        self.session.commit()
        logger.info(f"Successfully loaded {loaded_count} rulings")
        return loaded_count
    
    def load_rules(self, rules_file: Path) -> int:
        """Load comprehensive rules from rules text file."""
        logger.info(f"Loading rules from {rules_file}")
        
        with open(rules_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        loaded_count = 0
        current_section = None
        current_title = None
        
        # Parse rules by sections
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and non-rule content
            if not line or line.startswith('Magic:') or line.startswith('These rules') or line.startswith('Introduction'):
                i += 1
                continue
            
            # Check for section headers (e.g., "1. Game Concepts")
            section_match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if section_match:
                current_section = section_match.group(1)
                current_title = section_match.group(2)
                i += 1
                continue
            
            # Check for rule numbers (e.g., "100. General", "100.1", "100.1a")
            rule_match = re.match(r'^(\d+(?:\.\d+)*[a-z]?)\.\s+(.+)$', line)
            if rule_match:
                rule_id = rule_match.group(1)
                rule_content = rule_match.group(2)
                
                # Collect multi-line rule content
                i += 1
                while i < len(lines) and lines[i].strip() and not re.match(r'^\d+', lines[i].strip()):
                    rule_content += " " + lines[i].strip()
                    i += 1
                
                try:
                    # Check if rule already exists
                    existing = self.session.query(Rule).filter_by(id=rule_id).first()
                    if existing:
                        continue
                    
                    # Determine parent rule
                    parent_rule_id = None
                    if '.' in rule_id and len(rule_id.split('.')) > 1:
                        parts = rule_id.split('.')
                        if len(parts) > 2 or (len(parts) == 2 and parts[1][-1].isalpha()):
                            # This is a subrule
                            if parts[1][-1].isalpha():
                                parent_rule_id = f"{parts[0]}.{parts[1][:-1]}"
                            else:
                                parent_rule_id = parts[0]
                    
                    rule = Rule(
                        id=rule_id,
                        section=current_section or rule_id.split('.')[0],
                        title=current_title or "General",
                        content=rule_content,
                        parent_rule=parent_rule_id
                    )
                    
                    self.session.add(rule)
                    loaded_count += 1
                    
                    if loaded_count % 100 == 0:
                        self.session.commit()
                        logger.info(f"Loaded {loaded_count} rules...")
                        
                except Exception as e:
                    logger.error(f"Error loading rule {rule_id}: {e}")
                    continue
            else:
                i += 1
        
        self.session.commit()
        logger.info(f"Successfully loaded {loaded_count} rules")
        return loaded_count
    
    def load_all_data(self, data_dir: Path):
        """Load all data from the data directory."""
        cards_file = data_dir / "all_cards.json"
        rulings_file = data_dir / "rulings.json" 
        rules_file = data_dir / "rules"
        
        # Load cards first (rulings reference cards)
        if cards_file.exists():
            self.load_cards(cards_file)
        
        # Load rulings
        if rulings_file.exists():
            self.load_rulings(rulings_file)
        
        # Load rules
        if rules_file.exists():
            self.load_rules(rules_file)
    
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


if __name__ == "__main__":
    # Run data ingestion
    ingestion = DataIngestion()
    data_dir = Path("../../../")  # Adjust path as needed
    ingestion.load_all_data(data_dir)
    stats = ingestion.get_stats()
    print(f"Data loaded: {stats}")
    ingestion.close()