from sqlalchemy import Column, String, Text, DateTime, Float, Boolean, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Card(Base):
    __tablename__ = "cards"
    
    id = Column(String, primary_key=True)
    oracle_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=False, index=True)
    oracle_text = Column(Text)
    type_line = Column(String)
    mana_cost = Column(String)
    cmc = Column(Float)
    power = Column(String)
    toughness = Column(String)
    loyalty = Column(String)  # Starting loyalty for planeswalkers
    defense = Column(String)  # Defense value for battles
    colors = Column(JSON)
    color_identity = Column(JSON)
    keywords = Column(JSON)
    legalities = Column(JSON)
    set_code = Column(String)
    set_name = Column(String)
    rarity = Column(String)
    artist = Column(String)
    flavor_text = Column(Text)
    
    rulings = relationship("Ruling", back_populates="card")
    
    __table_args__ = (
        Index('idx_oracle_name', 'oracle_id', 'name'),
        Index('idx_name_search', 'name'),
    )


class Ruling(Base):
    __tablename__ = "rulings"
    
    id = Column(String, primary_key=True)
    oracle_id = Column(String, ForeignKey('cards.oracle_id'), nullable=False, index=True)
    source = Column(String, nullable=False)
    published_at = Column(DateTime, nullable=False)
    comment = Column(Text, nullable=False)
    
    card = relationship("Card", back_populates="rulings")


class Rule(Base):
    __tablename__ = "rules"
    
    id = Column(String, primary_key=True)  # e.g., "100.1a"
    section = Column(String, nullable=False, index=True)  # e.g., "100"
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    parent_rule = Column(String, ForeignKey('rules.id'))
    
    children = relationship("Rule")


class TrainingData(Base):
    __tablename__ = "training_data"
    
    id = Column(String, primary_key=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    is_correct = Column(Boolean, default=True)
    corrected_response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    feedback = Column(Text)
    
    __table_args__ = (
        Index('idx_query_search', 'query'),
        Index('idx_created_at', 'created_at'),
    )