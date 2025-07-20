# MTG Judge Chat Engine - Project Details

## Project Overview
A Magic: The Gathering judge chat engine that provides rulings and answers using local or remote LLMs with a web interface.

## Architecture
- **Backend**: Python with uv package management
- **LLM Integration**: Support for local (NVIDIA 4090) and remote LLMs
- **Framework**: DSPy for tool calling and rule processing
- **Web UI**: Interactive interface with card name typeahead
- **Database**: Local database for processed rules and rulings
- **Human Moderation**: Interface for correcting errors and improving training

## Data Sources
- Rules file (comprehensive MTG rules)
- Card rules text and rulings database
- Training data from moderated corrections

## Key Features
1. Natural language rule queries
2. Card-specific ruling lookups
3. Typeahead search for card names
4. Human moderation interface
5. Training data collection from corrections
6. Local/remote LLM flexibility

## Directory Structure
```
rulebot/
├── src/
│   ├── llm/           # LLM integration (local/remote)
│   ├── database/      # Rules and card data management
│   ├── dspy_tools/    # DSPy configuration and tools
│   ├── web/           # Web UI components
│   └── moderation/    # Human moderation interface
├── data/
│   ├── rules/         # MTG comprehensive rules
│   ├── cards/         # Card database and rulings
│   └── training/      # Training data from corrections
├── tests/
└── docs/
```

## Development Commands
- `uv run pytest` - Run tests
- `uv run uvicorn src.web.main:app --reload` - Start development server
- `uv run python -m src.database.setup` - Initialize database
- `uv run python -m src.llm.test_local` - Test local LLM connection

## LLM Setup
- Local: Configure for NVIDIA 4090 (CUDA support)
- Remote: API endpoints for cloud LLMs
- DSPy integration for structured prompting

## Data Placement Recommendations
- Place MTG comprehensive rules in: `data/rules/comprehensive_rules.txt`
- Place card database in: `data/cards/`
- Rulings data in: `data/cards/rulings/`
- Training corrections in: `data/training/`