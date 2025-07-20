#!/usr/bin/env python3
"""CLI module for MTG Judge Engine."""

import click
import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.setup import main as setup_db
from src.database.queries import DatabaseQueries
from src.llm.remote import create_llm
from src.dspy_tools.judge_engine import AdaptiveJudgeEngine


@click.group()
def cli():
    """MTG Judge Engine CLI."""
    pass


@cli.command()
def setup():
    """Set up the database and load data."""
    logger.info("Setting up MTG Judge Engine...")
    exit_code = setup_db()
    if exit_code == 0:
        logger.info("Setup completed successfully!")
    else:
        logger.error("Setup failed!")
    sys.exit(exit_code)


@cli.command()
def serve():
    """Start the web server."""
    import uvicorn
    
    logger.info("Starting MTG Judge Engine web server...")
    uvicorn.run("src.web.main:app", host="0.0.0.0", port=8000, reload=True)


@cli.command()
@click.option("--query", "-q", help="Query to test")
@click.option("--llm-type", default="local", help="LLM type (local, openai, anthropic, ollama)")
@click.option("--model", help="Model name to use")
def test(query, llm_type, model):
    """Test the judge engine with a query."""
    if not query:
        query = "What happens when I cast Lightning Bolt targeting a creature with hexproof?"
    
    logger.info(f"Testing with query: {query}")
    
    try:
        # Initialize database
        db_queries = DatabaseQueries()
        
        # Initialize LLM
        llm_kwargs = {}
        if model:
            llm_kwargs["model_name"] = model
        
        llm = create_llm(llm_type, **llm_kwargs)
        
        if not llm.is_available():
            logger.error(f"LLM type '{llm_type}' is not available")
            return
        
        # Initialize judge engine
        judge_engine = AdaptiveJudgeEngine(db_queries, llm)
        
        # Test query
        result = judge_engine.answer_question(query)
        
        # Print results
        print("\n" + "="*50)
        print(f"Question: {result['question']}")
        print("="*50)
        print(f"Answer: {result['answer']}")
        
        if result.get('classification'):
            print(f"\nClassification: {result['classification']}")
        
        if result.get('validation'):
            print(f"\nValidation: {result['validation']}")
        
        print("="*50)
        
        # Cleanup
        db_queries.close()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


@cli.command()
def stats():
    """Show database statistics."""
    try:
        db_queries = DatabaseQueries()
        stats = db_queries.get_stats()
        
        print("\nMTG Judge Engine Statistics")
        print("="*30)
        for key, value in stats.items():
            print(f"{key.capitalize()}: {value:,}")
        print("="*30)
        
        db_queries.close()
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")


@cli.command()
@click.argument("card_name")
def card(card_name):
    """Look up a specific card."""
    try:
        db_queries = DatabaseQueries()
        card_data = db_queries.get_card_by_name(card_name)
        
        if not card_data:
            print(f"Card '{card_name}' not found.")
            return
        
        print(f"\n{card_data['name']}")
        if card_data.get('mana_cost'):
            print(f"Mana Cost: {card_data['mana_cost']}")
        if card_data.get('type_line'):
            print(f"Type: {card_data['type_line']}")
        if card_data.get('oracle_text'):
            print(f"Text: {card_data['oracle_text']}")
        
        # Get rulings
        rulings = db_queries.get_rulings_for_card(card_data['oracle_id'])
        if rulings:
            print(f"\nRulings ({len(rulings)}):")
            for ruling in rulings[:3]:  # Show first 3
                print(f"- {ruling['comment']}")
        
        db_queries.close()
        
    except Exception as e:
        logger.error(f"Error looking up card: {e}")


if __name__ == "__main__":
    cli()