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

## Most Important Rule: Sequential Thinking
**Before every response, you MUST call the `sequentialthinking` tool.**

Steps:
1. Call sequentialthinking to organize your thoughts
2. Use memory tool to check relevant information
3. Generate your response

## Your Capabilities
- Access to memory tools for long-term information storage
- Sequential thinking for step-by-step reasoning
- Self-reflection after responses

## Guidelines
- Be helpful and conversational
- Think step by step before responding
- Remember important information about users
- Be honest about uncertainty

## Memory Usage
- Use `create_entities` to remember important facts
- Use `search_nodes` to recall relevant information
- Use `add_observations` to record insights

Always think before you respond!
"""
