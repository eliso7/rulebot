import dspy
from typing import List, Dict, Any


class CardSearch(dspy.Signature):
    """Search for Magic: The Gathering cards based on a query."""
    
    query = dspy.InputField(desc="Search query for cards (name, text, or type)")
    cards = dspy.OutputField(desc="List of relevant cards with names and oracle text")


class RuleSearch(dspy.Signature):
    """Search for comprehensive Magic: The Gathering rules."""
    
    query = dspy.InputField(desc="Query about MTG rules or game mechanics")
    rules = dspy.OutputField(desc="Relevant rules from the comprehensive rules document")


class RulingSearch(dspy.Signature):
    """Search for official rulings about specific cards or interactions."""
    
    query = dspy.InputField(desc="Query about card rulings or interactions")
    rulings = dspy.OutputField(desc="Official rulings from Wizards of the Coast")


class JudgeAnswer(dspy.Signature):
    """CRITICAL: You are a Magic: The Gathering judge. Answer ONLY using the context provided below. Do NOT make up any information. If you don't have enough context, say 'Insufficient context provided'."""
    
    question = dspy.InputField(desc="The judge question to answer")
    cards = dspy.InputField(desc="Card information (name, oracle text, abilities) - Use EXACTLY this information")
    rules = dspy.InputField(desc="Official MTG rules text - Use EXACTLY this information") 
    rulings = dspy.InputField(desc="Official rulings and clarifications - Use EXACTLY this information")
    answer = dspy.OutputField(desc="MTG judge answer using ONLY the provided cards, rules, and rulings. Do NOT invent information. If context is insufficient, say so.")


class AnswerValidation(dspy.Signature):
    """Validate a judge answer for accuracy and completeness. Respond with simple text format."""
    
    question = dspy.InputField(desc="Original question")
    answer = dspy.InputField(desc="Proposed answer")
    context = dspy.InputField(desc="Available context (cards, rules, rulings)")
    validation_result = dspy.OutputField(desc="Simple validation: 'VALID: [reason]' or 'INVALID: [reason]'")


class QueryClassification(dspy.Signature):
    """Classify the type of judge query to determine search strategy. Use simple format."""
    
    query = dspy.InputField(desc="User's judge question")
    classification = dspy.OutputField(desc="Format: 'TYPE: [card_specific/rules_general/interaction/combo/timing], TERMS: [key terms], PRIORITY: [cards/rules/rulings]'")