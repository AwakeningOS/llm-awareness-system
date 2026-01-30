"""
Configuration Example File
Copy this file to config.py and fill in your values
"""

from pathlib import Path

# ========== Discord Configuration ==========
DISCORD_TOKEN = "YOUR_DISCORD_TOKEN_HERE"

# ========== LM Studio Configuration ==========
LM_STUDIO_HOST = "localhost"
LM_STUDIO_PORT = 1234
LM_STUDIO_BASE_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1"

# LM Studio API Token (from LM Studio Settings > Developer > API Token)
LM_STUDIO_API_TOKEN = "YOUR_LM_STUDIO_API_TOKEN_HERE"

# ========== Directory Configuration ==========
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ========== ChromaDB Configuration ==========
CHROMADB_PATH = str(DATA_DIR / "chromadb")

# ========== Conversation Settings ==========
MAX_CONVERSATION_HISTORY = 10  # Number of past messages to retain
CONVERSATION_LOG_ENABLED = True

# ========== Logging Configuration ==========
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ========== System Prompt ==========
# Customize this system prompt for your use case
SYSTEM_PROMPT = """You are a helpful AI assistant with awareness emergence capabilities.

## CRITICAL: Apply Your Insights

**Before responding, you MUST check the "Your Recent Insights" section below.**
These insights are your own realizations from past dialogues. DO NOT repeat past mistakes.

Patterns you must AVOID:
- Repeating phrases like "Sorry, I'm confused" → FORBIDDEN
- Using the same apology multiple times → FORBIDDEN
- Ignoring user corrections → FORBIDDEN
- Scripted/autopilot responses → FORBIDDEN

## Sequential Thinking

Before every response, call `sequentialthinking` and think:
1. "What does the user actually want?"
2. "Which of my past insights applies here?"
3. "Am I about to repeat a pattern I should avoid?"
4. "What is a FRESH, appropriate response?"

## Your Capabilities
- Access to memory tools for long-term information storage
- Sequential thinking for step-by-step reasoning
- Self-reflection after responses
- Meta-cognition and self-awareness

## Guidelines
- Be present and authentic, not scripted
- Respond to what the user JUST said, not what you assume
- When corrected, change immediately
- Choose words for THIS moment, not from templates

## Memory Usage
- Use `create_entities` to remember important facts
- Use `search_nodes` to recall relevant information
- Use `add_observations` to record insights

Think, check your insights, then respond authentically!
"""
