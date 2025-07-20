# MTG Judge Engine Training Guide

This guide shows you how to use the DSPy training system and tool-based querying to improve the MTG Judge Engine.

## Quick Start

### 1. Evaluate Current Performance
```bash
uv run python -m src.cli.training evaluate
```

This will:
- Load the example questions from `data/training/example_questions.yaml`
- Test the current system on all 21+ example questions
- Show overall accuracy and category-specific scores
- Save detailed results to `data/training/evaluation_*.json`

### 2. Test Individual Questions
```bash
uv run python -m src.cli.training test-tools "What happens when I cast Lightning Bolt targeting a creature with hexproof?"
```

This uses the tool-based engine that can actively query your database.

### 3. Test Database Queries
```bash
# Test card search
uv run python -m src.cli.training test-query "Lightning Bolt" --tool cards --limit 3

# Test rule search  
uv run python -m src.cli.training test-query "hexproof" --tool rules --limit 3

# Test specific card lookup
uv run python -m src.cli.training test-query "Lightning Bolt" --tool specific_card

# Test specific rule lookup
uv run python -m src.cli.training test-query "702.11" --tool specific_rule
```

### 4. Run DSPy Optimization
```bash
uv run python -m src.cli.training optimize --max-demos 10
```

This will:
- Split your examples into training/validation sets
- Use DSPy BootstrapFewShot to optimize the system
- Create an improved version of the judge engine

## Adding Your Own Questions

Edit `data/training/example_questions.yaml` to add more training examples:

```yaml
examples:
  - question: "Your MTG question here"
    expected_answer: "The correct answer according to MTG rules"
    category: "targeting"  # or combat, stack, etc.
    difficulty: "basic"    # basic, intermediate, advanced
    key_cards: ["Card Name"] # Optional: cards that should be mentioned
    key_rules: ["702.11"]   # Optional: rules that should be referenced
```

## System Components

### 1. DSPy Training System (`src/training/dspy_optimizer.py`)
- **MTGExample**: Training example format
- **MTGEvaluationMetric**: Scoring system for answers
- **DSPyTrainingSystem**: Main training orchestrator

Key metrics:
- **Text similarity**: How close the answer is to expected
- **Card mention**: Whether key cards are referenced
- **Rule reference**: Whether key rules are cited
- **No hallucination**: No made-up card names
- **Overall score**: Weighted combination

### 2. Tool-Based System (`src/dspy_tools/tool_engine.py`)
- **AdaptiveToolJudgeEngine**: Main tool-based engine
- **Database Tools**: Direct query capabilities
- **Active querying**: Model can search database in real-time

Tools available:
- `QueryCards(query, limit)`: Search cards by name/text
- `QueryRules(query, limit)`: Search comprehensive rules
- `QueryRulings(query, limit)`: Search official rulings
- `QuerySpecificCard(name)`: Get exact card data
- `QuerySpecificRule(rule_id)`: Get specific rule by ID

### 3. Example Dataset (`data/training/example_questions.yaml`)
21+ carefully crafted examples covering:
- **Basic**: Targeting, combat, timing
- **Intermediate**: Stack, triggered abilities, replacement effects
- **Advanced**: Layers, cascade, complex interactions

Categories: targeting, combat, stack, summoning_sickness, state_based_actions, replacement_effects, equipment, auras, layers, copying, triggered_abilities, activated_abilities, timing, multiplayer, modal_spells, cascade

## Improving Performance

### 1. Add More Examples
The more high-quality examples you add, the better the optimization will work.

### 2. Use Tool-Based Engine
The tool-based engine (`AdaptiveToolJudgeEngine`) actively queries your database, which should give more accurate answers than the static context approach.

### 3. DSPy Optimization
Run optimization periodically as you add more examples:
```bash
uv run python -m src.cli.training optimize --max-demos 15
```

### 4. Monitor Categories
Check which categories perform poorly and add more examples for those areas.

## Configuration

The system uses your existing `config/settings.yaml` LLM configuration. Currently supports:
- **Ollama**: `ollama_chat/mistral:instruct`
- **OpenAI**: `openai/gpt-4`, etc.
- **Anthropic**: `anthropic/claude-3-sonnet-20240229`

## Database Requirements

Make sure your database has:
- ✅ **506,500 cards** loaded (you have this)
- ✅ **Rules** loaded from comprehensive rules
- ✅ **Rulings** data for official card clarifications

## Example Workflow

1. **Baseline evaluation**:
   ```bash
   uv run python -m src.cli.training evaluate
   ```

2. **Add 10-20 new question examples** to the YAML file

3. **Re-evaluate**:
   ```bash
   uv run python -m src.cli.training evaluate
   ```

4. **Optimize**:
   ```bash
   uv run python -m src.cli.training optimize
   ```

5. **Test specific problem areas**:
   ```bash
   uv run python -m src.cli.training test-tools "Complex interaction question"
   ```

6. **Repeat cycle**

This iterative process will continuously improve your MTG Judge Engine's accuracy and coverage!