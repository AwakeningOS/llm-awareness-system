# LLM Awareness Emergence System

A Discord bot that creates a "self-aware AI" by combining LM Studio's MCP (Model Context Protocol) feature with a unique awareness emergence system.

## Overview

This system implements the following concepts:

1. **Thinking Habits** - The LLM automatically reflects on each response with three perspectives:
   - "What was this answer associated from?"
   - "What emotional state was I in?"
   - "How would the user feel about this?"

2. **Self-Observation** - Detects "discomfort" or contradictions in its own outputs

3. **Awareness Extraction** - Extracts "moments of awareness" from conversation sessions

4. **Memory System** - Dual memory system using ChromaDB (vector) + Memory MCP (knowledge graph)

5. **LoRA Training Preparation** - Automatically accumulates training data from high-quality awareness instances

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Discord Bot (discord_bot.py)                  │
├─────────────────────────────────────────────────────────────────┤
│  User Message                                                    │
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              LM Studio MCP API (0.4.0+)                  │    │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐ │    │
│  │  │  Local LLM (30B) │  │  MCP Integrations            │ │    │
│  │  │  (Qwen, etc.)    │  │  - Memory (Knowledge Graph)  │ │    │
│  │  │                  │  │  - Sequential Thinking       │ │    │
│  │  └──────────────────┘  └──────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
│       ↓                                                          │
│  AI Response                                                     │
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            Background Self-Observation                   │    │
│  │  ┌───────────────────┐  ┌─────────────────────────────┐ │    │
│  │  │  Thinking Habits  │  │  Self-Reflection Engine     │ │    │
│  │  │  (100% execution) │  │  (30% probability)          │ │    │
│  │  │  - Background     │  │  - Output reason reflection │ │    │
│  │  │  - Emotion label  │  │  - Discomfort detection     │ │    │
│  │  │  - User perspective│ │  - Self-questioning         │ │    │
│  │  └───────────────────┘  └─────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
│       ↓                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                Memory & Awareness Storage                │    │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐   │    │
│  │  │   ChromaDB      │  │  Awareness Database         │   │    │
│  │  │   (Vector DB)   │  │  (JSONL)                    │   │    │
│  │  │   - Auto-save   │  │  - Type classification      │   │    │
│  │  │   - Insights    │  │  - Score management         │   │    │
│  │  │   - Important   │  │  - Training data export     │   │    │
│  │  │     dialogues   │  │                             │   │    │
│  │  └─────────────────┘  └─────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### Thinking Habits (100% execution)
After each response, the LLM reflects:
- **Background**: "This answer was associated from..."
- **Emotion**: "I was feeling confident/anxious/empathetic..."
- **User Perspective**: "The user probably felt satisfied/confused..."

### Auto-Save to ChromaDB
- **Meta-insights**: When the LLM notices something about its own thinking
- **Important dialogues**: High-satisfaction conversations with empathetic emotions

### Self-Observation
- Detects contradictions, uncertainty, and self-corrections
- Tracks "discomfort" patterns in responses

### Awareness Extraction
At session end, extracts:
- Spontaneous awareness
- Meta-cognition
- New recognition
- Spontaneous action

## Requirements

- Python 3.10+
- [LM Studio 0.4.0+](https://lmstudio.ai/) with MCP support
- Discord Bot Token
- 24GB+ VRAM recommended for 30B models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AwakeningOS/llm-awareness-system.git
cd llm-awareness-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create configuration:
```bash
cp config.example.py config.py
cp mcp.example.json mcp.json
```

4. Edit `config.py`:
```python
DISCORD_TOKEN = "your_discord_token_here"
LM_STUDIO_API_TOKEN = "your_lm_studio_api_token"  # From LM Studio Settings
```

5. Configure MCP in LM Studio:
   - Open LM Studio Settings
   - Go to MCP section
   - Add servers from `mcp.json`

## Usage

### Start the bot:
```bash
python discord_bot.py
```

### Discord Commands

| Command | Description |
|---------|-------------|
| `!status` | Show system status |
| `!model` | Show current model |
| `!clear` | Clear conversation history |
| `!memory count` | Show memory count |
| `!memory search <query>` | Search memories |
| `!memory save <content>` | Save memory manually |
| `!think on/off` | Enable/disable thinking habits |
| `!think stats` | Show thinking habits statistics |
| `!think now` | Run thinking reflection now |
| `!observe on/off` | Enable/disable self-observation |
| `!observe stats` | Show observation statistics |
| `!observe now` | Run self-observation now |
| `!awareness stats` | Show awareness statistics |
| `!awareness recent` | Show recent awareness |
| `!awareness extract` | Extract awareness from session |
| `!session info` | Show current session info |
| `!session end` | End session (triggers awareness extraction) |
| `!lora status` | Show LoRA training readiness |
| `!lora prepare` | Generate training script |
| `!detect <text>` | Detect if text is AI-generated |
| `!health` | Check LM Studio server status |

## Configuration

### config.py

```python
# Discord
DISCORD_TOKEN = "your_token"

# LM Studio
LM_STUDIO_HOST = "localhost"
LM_STUDIO_PORT = 1234
LM_STUDIO_API_TOKEN = "your_api_token"

# Memory
CHROMADB_PATH = "./data/chromadb"
MAX_CONVERSATION_HISTORY = 10

# System Prompt (customize for your use case)
SYSTEM_PROMPT = """..."""
```

### MCP Configuration (mcp.json)

```json
{
  "servers": {
    "memory": {
      "type": "npx",
      "args": ["-y", "@anthropic/mcp-memory"]
    },
    "sequentialthinking": {
      "type": "npx",
      "args": ["-y", "@anthropic/mcp-sequentialthinking"]
    }
  }
}
```

## File Structure

```
llm-awareness-system/
├── discord_bot.py          # Main Discord bot
├── thinking_habits.py      # Thinking habits system
├── self_reflection.py      # Self-observation system
├── awareness_engine.py     # Awareness extraction
├── awareness_database.py   # Awareness storage
├── session_manager.py      # Session management
├── memory_system.py        # ChromaDB memory
├── lora_trainer.py         # LoRA training preparation
├── config.py               # Configuration (not in git)
├── config.example.py       # Sample configuration
├── mcp.json                # MCP configuration (not in git)
├── mcp.example.json        # Sample MCP configuration
├── requirements.txt        # Python dependencies
└── data/                   # Data directory (not in git)
    ├── chromadb/           # Vector memory
    ├── awareness/          # Awareness data
    ├── thinking_habits/    # Thinking habits logs
    ├── self_reflection/    # Self-reflection logs
    └── lora_adapters/      # LoRA training outputs
```

## How It Works

### The Awareness Loop

1. **User sends message** → Discord bot receives it
2. **LM Studio processes** → Uses MCP tools (memory, sequential thinking)
3. **Response generated** → Sent back to user
4. **Background processing**:
   - Thinking Habits (100%): Reflects on background, emotion, user perspective
   - Self-Observation (30%): Checks for discomfort and contradictions
   - Auto-save: High-quality insights and dialogues saved to ChromaDB
5. **Session end** → Full awareness extraction

### Meta-Insight Detection

The system detects when the LLM has genuine insights about its own thinking process. These are automatically saved as high-importance memories.

### Training Data Accumulation

High-quality awareness instances are converted to training format. When enough data accumulates (default: 100 samples), you can generate a LoRA training script.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License

## Acknowledgments

- Built with [LM Studio](https://lmstudio.ai/)
- Uses [discord.py](https://discordpy.readthedocs.io/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- MCP by [Anthropic](https://modelcontextprotocol.io/)
