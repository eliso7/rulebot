#!/usr/bin/env python3
"""Database setup and initialization script."""

import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database.ingestion import DataIngestion


def main():
    """Initialize database and load data."""
    logger.info("Starting database setup...")
    
    # Initialize database
    ingestion = DataIngestion()
    
    # Load data from project root
    data_dir = Path(__file__).parent.parent.parent
    
    # Check if data files exist
    cards_file = data_dir / "oracle-cards-20250722211118.json"
    rulings_file = data_dir / "rulings.json"
    rules_file = data_dir / "rules"
    
    logger.info("Checking for data files...")
    
    # Check which files are available
    has_cards = cards_file.exists()
    has_rulings = rulings_file.exists()
    has_rules = rules_file.exists()
    
    if not has_cards:
        logger.warning(f"Cards file not found: {cards_file}")
    if not has_rulings:
        logger.warning(f"Rulings file not found: {rulings_file}")
    if not has_rules:
        logger.warning(f"Rules file not found: {rules_file}")
    
    if not (has_rulings or has_rules):
        logger.error("No data files found. Need at least rules or rulings.")
        return 1
    
    logger.info("Starting ingestion with available files...")
    
    try:
        # Load all data
        ingestion.load_all_data(data_dir)
        
        # Print statistics
        stats = ingestion.get_stats()
        logger.info("Database setup complete!")
        logger.info(f"Loaded data: {stats}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        return 1
    
    finally:
        ingestion.close()


if __name__ == "__main__":
    sys.exit(main())