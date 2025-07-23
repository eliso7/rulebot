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
from src.llm.qwen import QwenLLM
from src.judge.engine import MTGJudgeEngine
from src.rag.vector_store import FAISSVectorStore


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
@click.option("--model", default="Qwen/Qwen3-8B", help="Model name to use")
@click.option("--stream/--no-stream", default=True, help="Enable streaming output")
def test(query, model, stream):
    """Test the judge engine with a query."""
    if not query:
        query = "What happens when I cast Lightning Bolt targeting a creature with hexproof?"
    
    logger.info(f"Testing with query: {query}")
    
    try:
        # Initialize database
        db_queries = DatabaseQueries()
        
        # Initialize vector store
        vector_store = FAISSVectorStore()
        
        # Initialize Qwen LLM
        llm = QwenLLM(model_name=model)
        
        if not llm.is_available():
            logger.error(f"Qwen model '{model}' is not available")
            return
        
        # Initialize judge engine
        judge_engine = MTGJudgeEngine(llm, vector_store)
        
        # Test query
        result = judge_engine.answer_question(query, stream=stream)
        
        # Print results
        print("\n" + "="*50)
        print(f"Question: {result['question']}")
        print("="*50)
        print(f"Answer: {result['answer']}")
        
        if result.get('context_used'):
            print(f"\nContext Used: {len(result['context_used'])} sources")
        
        print("="*50)
        
        # Cleanup
        db_queries.close()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


@cli.command()
def populate():
    """Populate the vector store with data from the database."""
    logger.info("Populating vector store...")
    
    try:
        # Initialize database and vector store
        db_queries = DatabaseQueries()
        vector_store = FAISSVectorStore()
        
        # Clear existing data
        vector_store.clear()
        
        # Load rules
        logger.info("Loading rules...")
        rules = db_queries.get_all_rules()
        if rules:
            vector_store.add_rules(rules)
            logger.info(f"Added {len(rules)} rules")
        
        # Load cards (limit to avoid memory issues)
        logger.info("Loading cards...")
        cards = db_queries.get_all_cards(limit=10000)
        if cards:
            vector_store.add_cards(cards)
            logger.info(f"Added {len(cards)} cards")
        
        # Load rulings
        logger.info("Loading rulings...")
        rulings = db_queries.get_all_rulings(limit=5000)
        if rulings:
            vector_store.add_rulings(rulings)
            logger.info(f"Added {len(rulings)} rulings")
        
        # Save the index
        vector_store.save_index()
        
        # Show final stats
        stats = vector_store.get_stats()
        print(f"\nVector store populated successfully!")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Type breakdown: {stats['type_counts']}")
        
        db_queries.close()
        
    except Exception as e:
        logger.error(f"Error populating vector store: {e}")
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