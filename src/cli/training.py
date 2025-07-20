"""Command-line interface for training and testing the MTG Judge Engine."""

import click
from pathlib import Path
from loguru import logger

from ..config import get_config
from ..llm.remote import create_llm
from ..database.queries import DatabaseQueries
from ..training.dspy_optimizer import create_training_system, MTGExample
from ..dspy_tools.tool_engine import AdaptiveToolJudgeEngine


@click.group()
def training():
    """Training and evaluation commands for MTG Judge Engine."""
    pass


@training.command()
@click.option("--examples-file", default=None, help="Path to examples YAML file")
@click.option("--save-results", is_flag=True, default=True, help="Save evaluation results")
def evaluate(examples_file, save_results):
    """Evaluate current system performance on example questions."""
    try:
        # Setup
        config = get_config()
        llm_config = config.get_llm_config()
        llm_type = llm_config.get("type", "local")
        
        if llm_type == "ollama":
            ollama_config = llm_config.get("ollama", {})
            llm = create_llm("ollama", **ollama_config)
        elif llm_type == "openai":
            openai_config = llm_config.get("openai", {})
            llm = create_llm("openai", **openai_config)
        else:
            click.echo(f"Unsupported LLM type for evaluation: {llm_type}")
            return
        
        db_queries = DatabaseQueries()
        
        # Create training system
        training_system = create_training_system(llm, db_queries)
        
        # Load examples
        if examples_file:
            examples_path = Path(examples_file)
        else:
            examples_path = None
        
        examples = training_system.load_examples(examples_path)
        if not examples:
            click.echo("No examples loaded. Check your examples file.")
            return
        
        click.echo(f"Loaded {len(examples)} examples")
        
        # Run evaluation
        results = training_system.evaluate_current_system(examples)
        
        # Display results
        click.echo("\\n=== Evaluation Results ===")
        click.echo(f"Overall Average Score: {results['overall_average']:.3f}")
        click.echo("\\nCategory Scores:")
        for category, score in results['category_averages'].items():
            click.echo(f"  {category}: {score:.3f}")
        
        # Show some example results
        click.echo("\\n=== Sample Results ===")
        for i, result in enumerate(results['detailed_results'][:3]):
            click.echo(f"\\n{i+1}. Question: {result['question'][:80]}...")
            click.echo(f"   Expected: {result['expected'][:80]}...")
            click.echo(f"   Predicted: {result['predicted'][:80]}...")
            click.echo(f"   Score: {result['scores']['overall']:.3f}")
            click.echo(f"   Context: {result['context_used']['cards']} cards, {result['context_used']['rules']} rules")
        
        # Save results
        if save_results:
            training_system.save_evaluation_results(results)
            click.echo(f"\\nResults saved to data/training/evaluation_*.json")
        
        db_queries.close()
        
    except Exception as e:
        click.echo(f"Error during evaluation: {e}")
        logger.error(f"Evaluation error: {e}")


@training.command()
@click.option("--max-demos", default=20, help="Maximum number of demos for bootstrap")
@click.option("--examples-file", default=None, help="Path to examples YAML file")
@click.option("--intensive", is_flag=True, help="Run intensive multi-round training")
@click.option("--ultra", is_flag=True, help="Run ultra-aggressive training to 80% accuracy")
@click.option("--rounds", default=3, help="Number of training rounds for intensive mode")
@click.option("--target", default=0.8, help="Target accuracy for ultra mode")
@click.option("--timeout", default=30, help="Maximum time in minutes for ultra mode")
def optimize(max_demos, examples_file, intensive, ultra, rounds, target, timeout):
    """Optimize the judge engine using DSPy BootstrapFewShot."""
    try:
        # Setup
        config = get_config()
        llm_config = config.get_llm_config()
        llm_type = llm_config.get("type", "local")
        
        if llm_type == "ollama":
            ollama_config = llm_config.get("ollama", {})
            llm = create_llm("ollama", **ollama_config)
        elif llm_type == "openai":
            openai_config = llm_config.get("openai", {})
            llm = create_llm("openai", **openai_config)
        else:
            click.echo(f"Unsupported LLM type for optimization: {llm_type}")
            return
        
        db_queries = DatabaseQueries()
        training_system = create_training_system(llm, db_queries)
        
        # Load examples
        if examples_file:
            examples_path = Path(examples_file)
        else:
            examples_path = None
        
        examples = training_system.load_examples(examples_path)
        if not examples:
            click.echo("No examples loaded. Check your examples file.")
            return
        
        click.echo(f"Loaded {len(examples)} examples for optimization")
        
        if ultra:
            # Run ultra-aggressive training
            click.echo(f"Starting ULTRA-AGGRESSIVE training to reach {target:.1%} accuracy (max {timeout}min)...")
            optimized_engine = training_system.ultra_aggressive_training(target_score=target, max_time_minutes=timeout)
            
            if optimized_engine:
                click.echo("Ultra-aggressive training completed!")
                click.echo("Run 'evaluate' command to test the dramatically improved performance.")
            else:
                click.echo("Ultra-aggressive training failed to reach target.")
        elif intensive:
            # Run intensive multi-round training
            click.echo(f"Starting INTENSIVE training with {rounds} rounds...")
            optimized_engine = training_system.run_intensive_training(
                max_demos=max_demos, num_rounds=rounds
            )
            
            if optimized_engine:
                click.echo("Intensive training completed successfully!")
                click.echo("Run 'evaluate' command to test the improved performance.")
            else:
                click.echo("Intensive training failed.")
        else:
            # Regular single-round optimization
            # Split examples for training/validation
            split_point = int(len(examples) * 0.8)
            train_examples = examples[:split_point]
            val_examples = examples[split_point:]
            
            click.echo(f"Training on {len(train_examples)} examples, validating on {len(val_examples)}")
            
            # Run optimization
            click.echo("Starting DSPy optimization...")
            optimized_engine = training_system.optimize_with_bootstrap(
                train_examples, val_examples, max_demos
            )
            
            click.echo("Optimization completed!")
        
        click.echo("Note: Optimized engine is not automatically saved. Implement saving logic as needed.")
        
        db_queries.close()
        
    except Exception as e:
        click.echo(f"Error during optimization: {e}")
        logger.error(f"Optimization error: {e}")


@training.command()
@click.argument("question")
def test_tools(question):
    """Test the tool-based judge engine with a question."""
    try:
        # Setup
        config = get_config()
        llm_config = config.get_llm_config()
        llm_type = llm_config.get("type", "local")
        
        if llm_type == "ollama":
            ollama_config = llm_config.get("ollama", {})
            llm = create_llm("ollama", **ollama_config)
        elif llm_type == "openai":
            openai_config = llm_config.get("openai", {})
            llm = create_llm("openai", **openai_config)
        else:
            click.echo(f"Unsupported LLM type: {llm_type}")
            return
        
        db_queries = DatabaseQueries()
        
        # Create tool-based engine
        tool_engine = AdaptiveToolJudgeEngine(db_queries, llm)
        
        click.echo(f"Question: {question}")
        click.echo("Processing with tool-based engine...")
        
        # Get answer
        result = tool_engine.answer_question(question)
        
        # Display result
        click.echo("\\n=== Tool-Based Answer ===")
        click.echo(result.get('answer', 'No answer'))
        
        if 'query_results' in result:
            click.echo("\\n=== Query Results ===")
            for key, value in result['query_results'].items():
                click.echo(f"{key}: {len(value) if isinstance(value, str) and value.startswith('[') else 'N/A'} items")
        
        db_queries.close()
        
    except Exception as e:
        click.echo(f"Error testing tools: {e}")
        logger.error(f"Tool test error: {e}")


@training.command()
@click.argument("question")
@click.option("--tool", type=click.Choice(['cards', 'rules', 'rulings', 'specific_card', 'specific_rule']))
@click.option("--query", help="Query string for the tool")
@click.option("--limit", default=5, help="Limit for search results")
def test_query(question, tool, query, limit):
    """Test individual database query tools."""
    try:
        db_queries = DatabaseQueries()
        
        if not query:
            query = question
        
        click.echo(f"Testing {tool} tool with query: '{query}'")
        
        if tool == "cards":
            from ..dspy_tools.tool_engine import CardQueryTool
            tool_instance = CardQueryTool(db_queries)
            result = tool_instance(query, limit=limit, format_type='detailed')
            click.echo(result)
        
        elif tool == "rules":
            results = db_queries.search_rules(query, limit=limit)
            click.echo(f"Found {len(results)} rules:")
            for rule in results:
                click.echo(f"  - Rule {rule['id']}: {rule['content'][:60]}...")
        
        elif tool == "rulings":
            results = db_queries.search_rulings(query, limit=limit)
            click.echo(f"Found {len(results)} rulings:")
            for ruling in results:
                click.echo(f"  - {ruling.get('card_name', 'Unknown')}: {ruling['comment'][:60]}...")
        
        elif tool == "specific_card":
            from ..dspy_tools.tool_engine import SpecificCardTool
            tool_instance = SpecificCardTool(db_queries)
            result = tool_instance(query, format_type='detailed')
            click.echo(result)
        
        elif tool == "specific_rule":
            result = db_queries.get_rule_by_id(query)
            if result:
                click.echo(f"Found rule: {result['id']}")
                click.echo(f"Content: {result['content']}")
                
                subrules = db_queries.get_subrules(query)
                if subrules:
                    click.echo(f"Subrules: {len(subrules)}")
            else:
                click.echo(f"Rule '{query}' not found")
        
        db_queries.close()
        
    except Exception as e:
        click.echo(f"Error testing query: {e}")
        logger.error(f"Query test error: {e}")


if __name__ == "__main__":
    training()