# MTG Judge Chat Engine

A Magic: The Gathering judge chat engine that provides accurate rulings and rule interpretations using AI, with support for both local and remote LLMs.

## Features

- 🎯 **Accurate Rule Queries**: Ask questions about MTG rules in natural language
- 🃏 **Card-Specific Rulings**: Get rulings for specific cards with typeahead search
- 🤖 **Flexible LLM Support**: Works with local (NVIDIA GPU) or remote LLMs
- 🔧 **DSPy Integration**: Structured prompting and tool calling for better accuracy
- 🌐 **Web Interface**: Easy-to-use web UI for queries and results
- 👨‍⚖️ **Human Moderation**: Correct errors and improve the system over time
- 📚 **Comprehensive Data**: Built on official MTG rules and card databases

## Quick Start

### Prerequisites
- Python 3.11+
- uv package manager
- NVIDIA GPU (for local LLM) or API access to remote LLM

### Installation

1. Clone and navigate to the project:
```bash
cd rulebot
```

2. Install dependencies:
```bash
uv sync
```

3. Set up the database:
```bash
uv run python -m src.database.setup
```

4. Start the web interface:
```bash
uv run uvicorn src.web.main:app --reload
```

5. Open your browser to `http://localhost:8000`

## Usage

### Web Interface
1. Navigate to the web interface
2. Type your rule question or use typeahead to search for cards
3. Review the AI-generated response
4. Use moderation tools to correct any errors

### Example Queries
- "What happens when I cast Lightning Bolt targeting a creature with hexproof?"
- "Can I activate abilities of creatures with summoning sickness?"
- "How does cascade work with split cards?"

## Data Setup

Place your data files in these locations:
- **MTG Rules**: `data/rules/comprehensive_rules.txt`
- **Card Database**: `data/cards/`
- **Rulings**: `data/cards/rulings/`

## LLM Configuration

### Local LLM with bitsandbytes (NVIDIA GPU)

For optimal performance with limited GPU memory, install the GPU dependencies:

```bash
# Install GPU support with bitsandbytes for quantization
uv sync --extra gpu
```

This installs:
- `torch[cuda]` - PyTorch with CUDA support
- `accelerate` - Hugging Face acceleration library
- `bitsandbytes` - 4-bit/8-bit quantization for memory efficiency

**Hardware Requirements:**
- NVIDIA GPU with CUDA support (RTX 4090 recommended)
- At least 8GB VRAM for 7B models with 4-bit quantization
- 16GB+ VRAM for larger models

**Configuration in `config/settings.yaml`:**
```yaml
llm:
  type: "local"
  local:
    model_name: "mistralai/Mistral-7B-Instruct-v0.1"  # Recommended for MTG
    use_quantization: true  # Enables 4-bit quantization
    max_length: 2048
    temperature: 0.7
```

**Recommended Models:**
- `mistralai/Mistral-7B-Instruct-v0.1` - Best for rule interpretation
- `meta-llama/Llama-2-7b-chat-hf` - Good general purpose
- `codellama/CodeLlama-7b-Instruct-hf` - For complex rule logic

### Ollama Setup

Ollama provides an easy way to run local LLMs without GPU setup complexity.

**1. Install Ollama:**
```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai/download
```

**2. Pull a model:**
```bash
# Recommended for MTG rules
ollama pull llama2:7b-chat
ollama pull mistral:7b-instruct
ollama pull codellama:7b-instruct

# For better performance (requires more resources)
ollama pull llama2:13b-chat
ollama pull mistral:7b-instruct-v0.2
```

**3. Configure in `config/settings.yaml`:**
```yaml
llm:
  type: "ollama"
  ollama:
    model_name: "llama2:7b-chat"  # Or your preferred model
    base_url: "http://localhost:11434"
    temperature: 0.7
```

**4. Start Ollama service:**
```bash
# Ollama runs as a service after installation
# Verify it's running:
ollama list

# If not running, start it:
ollama serve
```

**Testing Ollama:**
```bash
# Test the model directly
ollama run llama2:7b-chat "What happens when I cast Lightning Bolt?"

# Test via the judge engine
uv run python -c "from src.llm.remote import OllamaLLM; llm = OllamaLLM(); print(llm.generate('Test').text)"
```

### Remote LLM APIs

For cloud-based inference, set API credentials:

**OpenAI:**
```bash
export OPENAI_API_KEY="your-api-key"
```

**Anthropic:**
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

**Configuration:**
```yaml
llm:
  type: "openai"  # or "anthropic"
  openai:
    model_name: "gpt-4"
    temperature: 0.7
    max_tokens: 1000
```

### Performance Comparison

| Setup | Memory | Speed | Quality | Cost |
|-------|--------|-------|---------|------|
| Local + bitsandbytes | 8-16GB VRAM | Fast | High | Free |
| Ollama | 8-32GB RAM | Medium | High | Free |
| OpenAI/Anthropic | Minimal | Very Fast | Highest | Paid |

## Development

### Running Tests
```bash
uv run pytest
```

### Code Style
```bash
uv run ruff check
uv run ruff format
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Wizards of the Coast for Magic: The Gathering
- DSPy framework for structured LLM interactions
- The MTG judge community for rule interpretations