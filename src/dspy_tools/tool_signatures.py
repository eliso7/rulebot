"""DSPy signatures for tool-based database querying."""

import dspy
from typing import List, Dict, Any


class QueryCards(dspy.Signature):
    """Search for Magic: The Gathering cards by name or text."""
    
    query = dspy.InputField(desc="Search query for card names or oracle text")
    limit = dspy.InputField(desc="Maximum number of cards to return (default: 5)")
    cards_json = dspy.OutputField(desc="JSON array of card objects with name, oracle_text, type_line, mana_cost")


class QueryRules(dspy.Signature):
    """Search the comprehensive Magic: The Gathering rules."""
    
    query = dspy.InputField(desc="Search query for rules content or keywords")
    limit = dspy.InputField(desc="Maximum number of rules to return (default: 5)")
    rules_json = dspy.OutputField(desc="JSON array of rule objects with id, content, section")


class QueryRulings(dspy.Signature):
    """Search official card rulings and clarifications."""
    
    query = dspy.InputField(desc="Search query for rulings or card interactions")
    limit = dspy.InputField(desc="Maximum number of rulings to return (default: 5)")
    rulings_json = dspy.OutputField(desc="JSON array of ruling objects with comment, card_name, published_at")


class QuerySpecificCard(dspy.Signature):
    """Get detailed information about a specific card by exact name."""
    
    card_name = dspy.InputField(desc="Exact name of the card to look up")
    include_rulings = dspy.InputField(desc="Whether to include rulings for this card (true/false)")
    card_data_json = dspy.OutputField(desc="JSON object with complete card data including oracle_text, rulings if requested")


class QuerySpecificRule(dspy.Signature):
    """Look up a specific rule by its number (e.g., '702.11' for hexproof)."""
    
    rule_number = dspy.InputField(desc="Rule number to look up (e.g., '702.11', '100.1')")
    rule_data_json = dspy.OutputField(desc="JSON object with rule id, content, section, and any subrules")


class ToolBasedJudgeAnswer(dspy.Signature):
    """CRITICAL: Answer as a Magic: The Gathering judge using ONLY the query results provided below. Do NOT make up information."""
    
    question = dspy.InputField(desc="The judge question to answer")
    available_tools = dspy.InputField(desc="Database query results with cards, rules, and rulings - USE ONLY THIS INFORMATION")
    answer = dspy.OutputField(desc="MTG judge answer using ONLY the provided query results. Quote specific cards and rules. Do NOT invent information.")


class PlanQueries(dspy.Signature):
    """Plan what database queries are needed to answer a judge question."""
    
    question = dspy.InputField(desc="The judge question to answer")
    query_plan = dspy.OutputField(desc="JSON array of query objects with tool_name, query_text, and reason")


class SynthesizeAnswer(dspy.Signature):
    """Synthesize a final answer from query results."""
    
    question = dspy.InputField(desc="Original judge question")
    query_results = dspy.InputField(desc="Results from database queries as JSON")
    answer = dspy.OutputField(desc="Final comprehensive answer based on the query results")