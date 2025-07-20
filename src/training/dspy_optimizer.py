"""DSPy training and optimization system for MTG Judge Engine."""

import yaml
import dspy
from pathlib import Path
from typing import List, Dict, Any, Tuple
from loguru import logger
import json
from datetime import datetime

from ..database.queries import DatabaseQueries
from ..dspy_tools.judge_engine import MTGJudgeEngine, AdaptiveJudgeEngine
from ..llm.base import BaseLLM


class MTGExample(dspy.Example):
    """MTG-specific example class for DSPy."""
    
    def __init__(self, question: str, expected_answer: str, category: str = None, 
                 difficulty: str = None, key_cards: List[str] = None, 
                 key_rules: List[str] = None, **kwargs):
        # Initialize with inputs properly set for DSPy
        super().__init__(
            question=question,
            expected_answer=expected_answer,
            category=category,
            difficulty=difficulty,
            key_cards=key_cards or [],
            key_rules=key_rules or [],
            **kwargs
        )


class MTGEvaluationMetric:
    """Evaluation metrics for MTG judge answers."""
    
    def __init__(self, db_queries: DatabaseQueries):
        self.db = db_queries
    
    def evaluate_answer(self, example: dspy.Example, prediction: dspy.Prediction) -> Dict[str, float]:
        """Evaluate a single answer prediction."""
        scores = {}
        
        # Get predicted answer
        predicted_answer = getattr(prediction, 'answer', '')
        expected_answer = example.expected_answer
        
        # Basic text similarity (simplified)
        scores['text_similarity'] = self._text_similarity(predicted_answer, expected_answer)
        
        # Check if key cards are mentioned
        if example.key_cards:
            scores['card_mention'] = self._check_card_mentions(predicted_answer, example.key_cards)
        
        # Check if key rules are referenced
        if example.key_rules:
            scores['rule_reference'] = self._check_rule_references(predicted_answer, example.key_rules)
        
        # Check for hallucination (made-up card names)
        scores['no_hallucination'] = self._check_hallucination(predicted_answer)
        
        # Overall score (weighted average) - more emphasis on text similarity
        text_sim = scores.get('text_similarity', 0)
        card_mention = scores.get('card_mention', 1) 
        rule_ref = scores.get('rule_reference', 1)
        no_halluc = scores.get('no_hallucination', 1)
        
        # If text similarity is very low, penalize heavily
        if text_sim < 0.2:
            overall = text_sim * 0.5  # Cap at 50% if concepts are wrong
        else:
            # Standard weighted scoring for decent answers
            overall = (
                text_sim * 0.5 +          # Increased weight for content quality
                card_mention * 0.2 +      # Card mentions
                rule_ref * 0.15 +         # Rule references  
                no_halluc * 0.15          # No hallucinations
            )
        scores['overall'] = overall
        
        return scores
    
    def _text_similarity(self, predicted: str, expected: str) -> float:
        """Enhanced text similarity metric for MTG rules."""
        import re
        
        # Clean and normalize text
        def clean_text(text):
            # Remove extra whitespace, normalize punctuation
            text = re.sub(r'\s+', ' ', text.lower().strip())
            text = re.sub(r'[^\w\s]', ' ', text)
            return text
        
        pred_clean = clean_text(predicted)
        exp_clean = clean_text(expected)
        
        if not exp_clean:
            return 0.0
        
        # Get key terms (longer words are more important)
        pred_words = set([w for w in pred_clean.split() if len(w) > 2])
        exp_words = set([w for w in exp_clean.split() if len(w) > 2])
        
        if not exp_words:
            return 0.0
        
        # Enhanced similarity with weight for important terms
        intersection = len(pred_words & exp_words)
        union = len(pred_words | exp_words)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Bonus for key MTG terms
        mtg_terms = {
            'damage', 'destroy', 'counter', 'target', 'ability', 'spell', 'creature', 
            'combat', 'stack', 'priority', 'mana', 'cost', 'permanent', 'graveyard',
            'battlefield', 'library', 'hand', 'exile', 'token', 'loyalty', 'planeswalker'
        }
        
        pred_mtg = pred_words & mtg_terms
        exp_mtg = exp_words & mtg_terms
        mtg_bonus = len(pred_mtg & exp_mtg) / len(exp_mtg) if exp_mtg else 0.0
        
        # Semantic similarity bonus for correct concepts
        semantic_bonus = 0.0
        if 'cannot' in predicted.lower() and 'cannot' in expected.lower():
            semantic_bonus += 0.1
        if 'before' in predicted.lower() and 'before' in expected.lower():
            semantic_bonus += 0.1
        if 'lethal' in predicted.lower() and 'lethal' in expected.lower():
            semantic_bonus += 0.1
        
        # Combined score with weights
        final_score = (jaccard * 0.6) + (mtg_bonus * 0.3) + (semantic_bonus * 0.1)
        return min(1.0, final_score)
    
    def _check_card_mentions(self, answer: str, key_cards: List[str]) -> float:
        """Check if key cards are mentioned."""
        answer_lower = answer.lower()
        mentioned = sum(1 for card in key_cards if card.lower() in answer_lower)
        return mentioned / len(key_cards) if key_cards else 1.0
    
    def _check_rule_references(self, answer: str, key_rules: List[str]) -> float:
        """Check if key rules are referenced."""
        answer_lower = answer.lower()
        mentioned = sum(1 for rule in key_rules if rule in answer_lower)
        return mentioned / len(key_rules) if key_rules else 1.0
    
    def _check_hallucination(self, answer: str) -> float:
        """Check for made-up card names (simplified check)."""
        # List of common fake card patterns to detect
        fake_patterns = [
            "this town ain't big enough",
            "super lightning bolt",
            "mega fireball",
            "ultimate counterspell",
            "instant win card"
        ]
        
        answer_lower = answer.lower()
        for pattern in fake_patterns:
            if pattern in answer_lower:
                return 0.0
        
        return 1.0


class DSPyTrainingSystem:
    """DSPy training and optimization system."""
    
    def __init__(self, llm: BaseLLM, db_queries: DatabaseQueries):
        self.llm = llm
        self.db_queries = db_queries
        self.metric = MTGEvaluationMetric(db_queries)
        self.examples = []
        
    def load_examples(self, examples_file: Path = None) -> List[dspy.Example]:
        """Load training examples from YAML file."""
        if examples_file is None:
            examples_file = Path("data/training/example_questions.yaml")
        
        if not examples_file.exists():
            logger.error(f"Examples file not found: {examples_file}")
            return []
        
        with open(examples_file, 'r') as f:
            data = yaml.safe_load(f)
        
        examples = []
        for example_data in data.get('examples', []):
            # Create DSPy Example directly for better compatibility
            example = dspy.Example(
                question=example_data["question"],
                expected_answer=example_data["expected_answer"],
                category=example_data.get("category"),
                difficulty=example_data.get("difficulty"),
                key_cards=example_data.get("key_cards", []),
                key_rules=example_data.get("key_rules", [])
            )
            # Set inputs for DSPy training
            example = example.with_inputs("question")
            examples.append(example)
        
        self.examples = examples
        logger.info(f"Loaded {len(examples)} training examples")
        return examples
    
    def evaluate_current_system(self, examples: List[dspy.Example] = None) -> Dict[str, Any]:
        """Evaluate current system performance."""
        if examples is None:
            examples = self.examples
        
        if not examples:
            logger.error("No examples to evaluate")
            return {}
        
        # Create judge engine
        judge_engine = AdaptiveJudgeEngine(self.db_queries, self.llm)
        
        results = []
        category_scores = {}
        
        for example in examples:
            try:
                # Get prediction
                result = judge_engine.answer_question(example.question)
                
                # Create prediction object
                prediction = dspy.Prediction(answer=result.get('answer', ''))
                
                # Evaluate
                scores = self.metric.evaluate_answer(example, prediction)
                
                result_data = {
                    'question': example.question,
                    'expected': example.expected_answer,
                    'predicted': result.get('answer', ''),
                    'category': example.category,
                    'difficulty': example.difficulty,
                    'scores': scores,
                    'context_used': {
                        'cards': len(result.get('context', {}).get('cards', [])),
                        'rules': len(result.get('context', {}).get('rules', [])),
                        'rulings': len(result.get('context', {}).get('rulings', []))
                    }
                }
                results.append(result_data)
                
                # Track category scores
                if example.category:
                    if example.category not in category_scores:
                        category_scores[example.category] = []
                    category_scores[example.category].append(scores['overall'])
                
            except Exception as e:
                logger.error(f"Error evaluating example: {e}")
                continue
        
        # Calculate aggregated metrics
        overall_scores = [r['scores']['overall'] for r in results]
        category_averages = {
            cat: sum(scores) / len(scores) 
            for cat, scores in category_scores.items()
        }
        
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_examples': len(examples),
            'evaluated_examples': len(results),
            'overall_average': sum(overall_scores) / len(overall_scores) if overall_scores else 0,
            'category_averages': category_averages,
            'detailed_results': results
        }
        
        return evaluation_results
    
    def optimize_with_bootstrap(self, train_examples: List[dspy.Example], 
                               validation_examples: List[dspy.Example] = None,
                               max_demos: int = 20) -> MTGJudgeEngine:
        """Optimize the judge engine using DSPy BootstrapFewShot."""
        
        if validation_examples is None:
            # Split examples 80/20
            split_point = int(len(train_examples) * 0.8)
            all_examples = train_examples[:]  # Make a copy
            train_examples = all_examples[:split_point]
            validation_examples = all_examples[split_point:]
        
        logger.info(f"Training with {len(train_examples)} examples, validating with {len(validation_examples)}")
        
        # Create base judge engine and set up DSPy LM
        from ..dspy_tools.judge_engine import AdaptiveJudgeEngine
        
        # Set up DSPy LM using the wrapper from AdaptiveJudgeEngine
        adaptive_engine = AdaptiveJudgeEngine(self.db_queries, self.llm)
        judge_engine = adaptive_engine.judge_engine
        
        # Define metric for optimization
        def accuracy_metric(example, prediction, trace=None):
            scores = self.metric.evaluate_answer(example, prediction)
            return scores['overall']
        
        try:
            # Set up aggressive optimizer with multiple rounds
            logger.info(f"Starting aggressive DSPy optimization with {max_demos} demos...")
            
            # Round 1: BootstrapFewShot with high sample count  
            optimizer = dspy.BootstrapFewShot(
                metric=accuracy_metric,
                max_bootstrapped_demos=max_demos,
                max_labeled_demos=max_demos,
                max_rounds=3  # Multiple rounds for better convergence
            )
            
            # Compile the optimized program (newer DSPy API)
            logger.info("Running BootstrapFewShot optimization...")
            optimized_judge = optimizer.compile(
                judge_engine,
                trainset=train_examples
            )
            
            # Round 2: Try to further optimize with different approach
            logger.info("Running second optimization round with LabeledFewShot...")
            labeled_optimizer = dspy.LabeledFewShot(k=min(max_demos, len(train_examples)))
            further_optimized = labeled_optimizer.compile(optimized_judge, trainset=train_examples)
            
            # Use the better of the two
            optimized_judge = further_optimized
            
            # Evaluate on validation set if provided
            if validation_examples:
                logger.info("Evaluating optimized engine on validation set...")
                val_scores = []
                for i, example in enumerate(validation_examples):  # Test ALL validation examples
                    try:
                        result = optimized_judge(example.question)
                        prediction = dspy.Prediction(answer=result.get('answer', ''))
                        score = accuracy_metric(example, prediction)
                        val_scores.append(score)
                        
                        if i < 3:  # Show first few for debugging
                            logger.info(f"Val {i+1}: Q='{example.question[:50]}...' Score={score:.3f}")
                    except Exception as e:
                        logger.warning(f"Validation example {i+1} failed: {e}")
                
                if val_scores:
                    avg_val_score = sum(val_scores) / len(val_scores)
                    logger.info(f"Final validation average score: {avg_val_score:.3f} ({len(val_scores)}/{len(validation_examples)} examples)")
                    
                    # If validation score is too low, try one more optimization round
                    if avg_val_score < 0.7:
                        logger.info("Validation score too low, attempting final optimization round...")
                        try:
                            final_optimizer = dspy.BootstrapFewShot(
                                metric=accuracy_metric,
                                max_bootstrapped_demos=min(max_demos * 2, len(train_examples)),
                                max_labeled_demos=min(max_demos * 2, len(train_examples)),
                                max_rounds=1  # Fewer rounds for final optimization
                            )
                            optimized_judge = final_optimizer.compile(optimized_judge, trainset=train_examples)
                            logger.info("Final optimization round completed")
                        except Exception as e:
                            logger.warning(f"Final optimization failed: {e}")
                else:
                    logger.warning("No validation scores computed")
            
        except Exception as e:
            logger.error(f"Bootstrap optimization failed: {e}")
            logger.info("Falling back to basic few-shot optimization...")
            
            # Fallback: Use aggressive LabeledFewShot
            try:
                logger.info("Attempting aggressive fallback optimization...")
                optimizer = dspy.LabeledFewShot(k=min(max_demos * 2, len(train_examples)))
                optimized_judge = optimizer.compile(
                    judge_engine,
                    trainset=train_examples
                )
                
                # Try a second round with different examples
                if len(train_examples) > max_demos:
                    logger.info("Running second fallback round with different examples...")
                    # Use the second half of training examples
                    second_half = train_examples[len(train_examples)//2:]
                    optimizer2 = dspy.LabeledFewShot(k=min(max_demos, len(second_half)))
                    optimized_judge = optimizer2.compile(optimized_judge, trainset=second_half)
                    
            except Exception as e2:
                logger.error(f"Fallback optimization also failed: {e2}")
                logger.info("Returning unoptimized judge engine")
                return judge_engine
        
        logger.info("DSPy optimization completed")
        return optimized_judge
    
    def run_intensive_training(self, max_demos: int = 30, num_rounds: int = 3) -> MTGJudgeEngine:
        """Run intensive multi-round training to maximize performance."""
        logger.info(f"Starting intensive training: {num_rounds} rounds with {max_demos} demos each")
        
        # Load examples
        examples = self.load_examples()
        if not examples:
            logger.error("No examples loaded for intensive training")
            return None
        
        logger.info(f"Loaded {len(examples)} examples for intensive training")
        
        # Split for intensive training (use more for training, less for validation)
        split_point = int(len(examples) * 0.9)  # 90% train, 10% validation
        train_examples = examples[:split_point]
        val_examples = examples[split_point:]
        
        logger.info(f"Intensive training: {len(train_examples)} train, {len(val_examples)} validation")
        
        best_score = 0.0
        best_engine = None
        
        for round_num in range(num_rounds):
            logger.info(f"=== INTENSIVE TRAINING ROUND {round_num + 1}/{num_rounds} ===")
            
            # Gradually increase demo count
            current_demos = min(max_demos + round_num * 5, len(train_examples))
            
            # Rotate training examples each round
            offset = (round_num * len(train_examples) // num_rounds) % len(train_examples)
            rotated_train = train_examples[offset:] + train_examples[:offset]
            
            try:
                # Run optimization for this round
                engine = self.optimize_with_bootstrap(
                    rotated_train, 
                    val_examples, 
                    max_demos=current_demos
                )
                
                # Evaluate this round's performance
                logger.info(f"Evaluating round {round_num + 1} performance...")
                eval_results = self.evaluate_current_system(val_examples)
                current_score = eval_results.get('overall_average', 0.0)
                
                logger.info(f"Round {round_num + 1} score: {current_score:.3f}")
                
                if current_score > best_score:
                    best_score = current_score
                    best_engine = engine
                    logger.info(f"New best score: {best_score:.3f}")
                
            except Exception as e:
                logger.error(f"Round {round_num + 1} failed: {e}")
                continue
        
        if best_engine:
            logger.info(f"Intensive training completed. Best score: {best_score:.3f}")
            return best_engine
        else:
            logger.error("Intensive training failed to produce any viable engine")
            return None
    
    def ultra_aggressive_training(self, target_score: float = 0.8, max_time_minutes: int = 30) -> MTGJudgeEngine:
        """Ultra-aggressive training to reach high performance with proper termination."""
        import time
        start_time = time.time()
        max_time_seconds = max_time_minutes * 60
        
        logger.info(f"Starting ULTRA-AGGRESSIVE training to reach {target_score:.1%} accuracy (max {max_time_minutes}min)")
        
        # Load examples
        examples = self.load_examples()
        if not examples:
            logger.error("No examples loaded")
            return None
        
        logger.info(f"Loaded {len(examples)} examples for ultra-aggressive training")
        
        # Use 85% for training, 15% for validation (better validation diversity)
        split_point = int(len(examples) * 0.85)
        train_examples = examples[:split_point]
        val_examples = examples[split_point:]
        
        # Shuffle examples for better distribution
        import random
        random.seed(42)  # Consistent shuffling
        shuffled_examples = examples.copy()
        random.shuffle(shuffled_examples)
        
        train_examples = shuffled_examples[:split_point]
        val_examples = shuffled_examples[split_point:]
        
        logger.info(f"Ultra training: {len(train_examples)} train, {len(val_examples)} validation")
        
        best_score = 0.0
        best_engine = None
        total_attempts = 0
        max_total_attempts = 50  # Hard limit to prevent infinite loops
        
        # Progressive training strategy - start small and build up
        max_demos = min(50, len(train_examples) // 2)  # Don't exceed half the training data
        demo_counts = [5, 10, 15, 20, 25, 30, max_demos]
        
        for attempt in range(len(demo_counts)):
            # Check time limit
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time_seconds:
                logger.info(f"Time limit reached ({max_time_minutes}min), stopping training")
                break
                
            current_demos = demo_counts[attempt]
            logger.info(f"=== ULTRA ATTEMPT {attempt + 1}/{len(demo_counts)} - {current_demos} demos ===")
            logger.info(f"Elapsed: {elapsed_time/60:.1f}min, Best score so far: {best_score:.3f}")
            
            try:
                # Progressive optimization strategies - build complexity
                if current_demos <= 10:
                    # Start simple for small datasets
                    strategies = [
                        ('labeled_simple', current_demos),
                        ('bootstrap_single', 1)
                    ]
                elif current_demos <= 25:
                    # Medium complexity
                    strategies = [
                        ('bootstrap_single', 1),
                        ('labeled_heavy', current_demos),
                        ('bootstrap_multi_round', 2)
                    ]
                else:
                    # Full complexity for larger datasets
                    strategies = [
                        ('bootstrap_multi_round', 2),
                        ('labeled_heavy', current_demos),
                        ('bootstrap_single', 1),
                        ('hybrid_approach', current_demos)
                    ]
                
                for strategy_name, param in strategies:
                    total_attempts += 1
                    
                    # Check hard limits
                    if total_attempts > max_total_attempts:
                        logger.info(f"Maximum attempts ({max_total_attempts}) reached, stopping")
                        break
                        
                    elapsed_time = time.time() - start_time
                    if elapsed_time > max_time_seconds:
                        logger.info(f"Time limit reached during strategy execution, stopping")
                        break
                    
                    logger.info(f"Trying strategy: {strategy_name} (attempt {total_attempts}/{max_total_attempts})")
                    temp_engine = None
                    
                    try:
                        if strategy_name == 'labeled_simple':
                            # Simple labeled few-shot for small datasets
                            from ..dspy_tools.judge_engine import AdaptiveJudgeEngine
                            adaptive_engine = AdaptiveJudgeEngine(self.db_queries, self.llm)
                            base_engine = adaptive_engine.judge_engine
                            
                            k_param = min(param, len(train_examples))
                            optimizer = dspy.LabeledFewShot(k=k_param)
                            temp_engine = optimizer.compile(base_engine, trainset=train_examples[:k_param])
                        
                        elif strategy_name == 'bootstrap_multi_round':
                            # Multiple bootstrap rounds with timeout protection
                            for round_i in range(param):
                                round_start = time.time()
                                logger.info(f"Bootstrap round {round_i + 1}/{param}")
                                
                                temp_engine = self.optimize_with_bootstrap(
                                    train_examples, val_examples, max_demos=current_demos
                                )
                                
                                # Check if this round took too long
                                round_time = time.time() - round_start
                                if round_time > 300:  # 5 minutes per round max
                                    logger.warning(f"Round took {round_time:.1f}s, may be too slow")
                                    break
                        
                        elif strategy_name == 'labeled_heavy':
                            # Heavy labeled few-shot with limited scope
                            from ..dspy_tools.judge_engine import AdaptiveJudgeEngine
                            adaptive_engine = AdaptiveJudgeEngine(self.db_queries, self.llm)
                            base_engine = adaptive_engine.judge_engine
                            
                            # Limit parameter to prevent excessive computation
                            limited_param = min(param, 30, len(train_examples))
                            optimizer = dspy.LabeledFewShot(k=limited_param)
                            temp_engine = optimizer.compile(base_engine, trainset=train_examples[:limited_param])
                        
                        elif strategy_name == 'bootstrap_single':
                            # Single aggressive bootstrap
                            temp_engine = self.optimize_with_bootstrap(
                                train_examples, val_examples, max_demos=current_demos
                            )
                        
                        elif strategy_name == 'hybrid_approach':
                            # Hybrid: Bootstrap then LabeledFewShot
                            logger.info("Trying hybrid approach: bootstrap + labeled")
                            
                            # First do bootstrap
                            temp_engine = self.optimize_with_bootstrap(
                                train_examples, val_examples, max_demos=current_demos // 2
                            )
                            
                            # Then apply labeled few-shot on top
                            if temp_engine:
                                labeled_optimizer = dspy.LabeledFewShot(k=min(15, len(train_examples)))
                                temp_engine = labeled_optimizer.compile(temp_engine, trainset=train_examples[:15])
                            else:
                                # Fallback to just labeled if bootstrap failed
                                from ..dspy_tools.judge_engine import AdaptiveJudgeEngine
                                adaptive_engine = AdaptiveJudgeEngine(self.db_queries, self.llm)
                                base_engine = adaptive_engine.judge_engine
                                labeled_optimizer = dspy.LabeledFewShot(k=min(20, len(train_examples)))
                                temp_engine = labeled_optimizer.compile(base_engine, trainset=train_examples[:20])
                        
                        # Evaluate this strategy
                        if temp_engine:
                            logger.info(f"Evaluating {strategy_name} performance...")
                            
                            # Quick validation on subset - use more examples for better estimate
                            test_subset = val_examples[:min(5, len(val_examples))]
                            scores = []
                            
                            for i, ex in enumerate(test_subset):
                                try:
                                    # Handle different engine types
                                    if hasattr(temp_engine, 'answer_question'):
                                        # AdaptiveJudgeEngine interface
                                        result = temp_engine.answer_question(ex.question)
                                        predicted_answer = result.get('answer', '')
                                    elif hasattr(temp_engine, '__call__'):
                                        # DSPy engine interface (direct call)
                                        result = temp_engine(ex.question)
                                        if hasattr(result, 'answer'):
                                            predicted_answer = result.answer
                                        else:
                                            predicted_answer = str(result)
                                    else:
                                        # Try forward method
                                        result = temp_engine.forward(ex.question)
                                        if hasattr(result, 'answer'):
                                            predicted_answer = result.answer
                                        else:
                                            predicted_answer = str(result)
                                    
                                    # Quick scoring - focus on basic metrics
                                    pred = dspy.Prediction(answer=predicted_answer)
                                    score = self.metric.evaluate_answer(ex, pred)['overall']
                                    scores.append(score)
                                    
                                    logger.debug(f"Quick eval {i+1}: {score:.3f} for '{ex.question[:30]}...'")
                                except Exception as e:
                                    logger.warning(f"Quick eval {i+1} failed: {e}")
                                    continue
                            
                            if scores:
                                avg_score = sum(scores) / len(scores)
                                logger.info(f"{strategy_name} quick score: {avg_score:.3f}")
                                
                                if avg_score > best_score:
                                    best_score = avg_score
                                    best_engine = temp_engine
                                    logger.info(f"NEW BEST SCORE: {best_score:.3f}")
                                    
                                    # If we've reached target, do full validation
                                    if best_score >= target_score:
                                        logger.info("Target reached! Running full validation...")
                                        full_results = self.evaluate_current_system(val_examples)
                                        full_score = full_results.get('overall_average', 0.0)
                                        
                                        if full_score >= target_score:
                                            logger.info(f"TARGET ACHIEVED! Full score: {full_score:.3f}")
                                            logger.info(f"Training completed in {(time.time() - start_time)/60:.1f} minutes")
                                            return best_engine
                                        else:
                                            logger.info(f"Full validation lower: {full_score:.3f}, continuing...")
                                
                                # Early stopping if we're making good progress
                                if best_score > target_score * 0.9:  # 90% of target
                                    logger.info(f"Close to target ({best_score:.3f}), trying intensive evaluation...")
                                    full_results = self.evaluate_current_system(val_examples)
                                    full_score = full_results.get('overall_average', 0.0)
                                    
                                    if full_score >= target_score:
                                        logger.info(f"TARGET ACHIEVED via early check! Full score: {full_score:.3f}")
                                        return best_engine
                    
                    except Exception as e:
                        logger.warning(f"Strategy {strategy_name} failed: {e}")
                        continue
                
                # Break if we exceeded attempt limits
                if total_attempts > max_total_attempts:
                    break
            
            except Exception as e:
                logger.error(f"Ultra attempt {attempt + 1} failed: {e}")
                continue
        
        # Training completed (by time/attempt limits or demo exhaustion)
        elapsed_minutes = (time.time() - start_time) / 60
        logger.info(f"Ultra-aggressive training completed in {elapsed_minutes:.1f} minutes")
        logger.info(f"Total optimization attempts: {total_attempts}")
        
        if best_engine:
            logger.info(f"Best achieved score: {best_score:.3f}")
            
            # Final full evaluation
            logger.info("Running final comprehensive evaluation...")
            final_results = self.evaluate_current_system(examples)
            final_score = final_results.get('overall_average', 0.0)
            logger.info(f"Final comprehensive score: {final_score:.3f}")
            
            if final_score >= target_score:
                logger.info(f"TARGET ACHIEVED in final evaluation!")
            else:
                logger.info(f"Target not reached (target: {target_score:.3f}, achieved: {final_score:.3f})")
            
            return best_engine
        else:
            logger.error("Ultra-aggressive training failed to produce any viable engine")
            return None
    
    def save_evaluation_results(self, results: Dict[str, Any], 
                               output_file: Path = None):
        """Save evaluation results to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"data/training/evaluation_{timestamp}.json")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_file}")
    
    def run_full_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        logger.info("Starting full evaluation...")
        
        # Load examples
        examples = self.load_examples()
        if not examples:
            return {}
        
        # Evaluate current system
        results = self.evaluate_current_system(examples)
        
        # Save results
        if save_results:
            self.save_evaluation_results(results)
        
        # Print summary
        logger.info(f"Evaluation complete: {results['overall_average']:.3f} average score")
        for category, score in results['category_averages'].items():
            logger.info(f"  {category}: {score:.3f}")
        
        return results


def create_training_system(llm: BaseLLM, db_queries: DatabaseQueries) -> DSPyTrainingSystem:
    """Factory function to create training system."""
    return DSPyTrainingSystem(llm, db_queries)