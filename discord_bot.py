"""
Discord Bot - MCP Enabled + Awareness Emergence System
Uses LM Studio's MCP feature for simple implementation

Improvements:
- TTL settings for optimized memory management
- Short-form integrations support
- Detailed tool call logging
- Server health check functionality
- Extracts reasoning/tool_calls from responses
- Integrated awareness emergence system
"""

import discord
from discord.ext import commands
from openai import OpenAI
from datetime import datetime
from pathlib import Path
import asyncio
import logging
import requests
import json

from config import (
    DISCORD_TOKEN,
    LM_STUDIO_HOST,
    LM_STUDIO_PORT,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_TOKEN,
    CHROMADB_PATH,
    MAX_CONVERSATION_HISTORY,
    CONVERSATION_LOG_ENABLED,
    SYSTEM_PROMPT,
    LOGS_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
    DATA_DIR,
    DREAMING_CONFIG,
)
from memory_system import MemorySystem
from awareness_engine import AwarenessEngine, AITextDetector
from session_manager import SessionManager, Session
from awareness_database import AwarenessDatabase
from lora_trainer import LoRATrainer, TrainingNotifier
from self_reflection import SelfReflectionEngine, RealtimeObserver
from thinking_habits import ThinkingHabitsEngine, RealtimeThinkingHabits
from dreaming_engine import DreamingEngine

# Logging configuration
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# ========== LM Studio 0.4.0 Configuration ==========
# MCP Configuration
MCP_INTEGRATIONS = [
    "mcp/memory",  # Short form (recommended in documentation)
    "mcp/sequential-thinking",  # Sequential thinking for structured reasoning
    # For future additions:
    # "mcp/playwright",
    # {
    #     "type": "ephemeral_mcp",
    #     "server_label": "huggingface",
    #     "server_url": "https://huggingface.co/mcp",
    #     "allowed_tools": ["model_search"]
    # }
]

# Model TTL setting (seconds) - auto-unload when idle
MODEL_TTL = 1800  # 30 minutes

# Context length
CONTEXT_LENGTH = 8400

# ========== Client Initialization ==========
# LM Studio OpenAI-compatible client (fallback)
llm_client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_TOKEN
)

# LM Studio API endpoints
LM_STUDIO_MCP_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/api/v1/chat"
LM_STUDIO_MODELS_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/api/v1/models"

# ChromaDB memory system
memory = MemorySystem(data_dir=CHROMADB_PATH)

# ========== Awareness Emergence System ==========
# Awareness engine
awareness_engine = AwarenessEngine()

# AI text detector
ai_detector = AITextDetector()

# Awareness database
awareness_db = AwarenessDatabase(data_dir=str(DATA_DIR / "awareness"))

# LoRA trainer
lora_trainer = LoRATrainer(awareness_db, output_dir=str(DATA_DIR / "lora_adapters"))

# Training notifier
training_notifier = TrainingNotifier(lora_trainer)

# ========== Self-Observation Enhancement System ==========
# Self-reflection engine
self_reflection_engine = SelfReflectionEngine()

# Realtime observer (100% probability - FULL OPEN MODE)
realtime_observer = RealtimeObserver(
    self_reflection_engine,
    observation_probability=1.0,  # 100% - always observe
    always_detect_discomfort=True
)

# Self-observation enabled flag (per user) - Default ON
self_observation_enabled: dict[str, bool] = {}
SELF_OBSERVATION_DEFAULT = True

# ========== Thinking Habits System ==========
# Thinking habits engine
thinking_habits_engine = ThinkingHabitsEngine()

# Realtime thinking habits (100% reflection probability)
realtime_thinking = RealtimeThinkingHabits(
    thinking_habits_engine,
    reflection_probability=1.0  # 100% execution
)

# Thinking habits enabled flag (per user) - Default ON
thinking_habits_enabled: dict[str, bool] = {}
THINKING_HABITS_DEFAULT = True

# ========== Dreaming Time System ==========
# Dreaming engine (uses global memory)
dreaming_engine = DreamingEngine(memory)

# Dreaming state
is_dreaming = False
dream_notification_channel = None
last_dream_memory_count = 0  # Memory count at last dream completion (for cooldown)


# Session end callback - DISABLED
# Awareness extraction moved to Dreaming Time system
# Dialogues are now saved immediately to ChromaDB
async def on_session_end(session: Session):
    """Session end callback - now just logs, no extraction"""
    logger.info(f"Session ended (awareness extraction disabled): {session.user_id}")
    # Awareness extraction is now handled by Dreaming Time
    # Dialogues are saved immediately after each exchange


# Session manager (simplified - no auto-extraction)
session_manager = SessionManager(
    timeout_seconds=1800,  # 30 minutes (just for session grouping)
    on_session_end=on_session_end,
    session_log_dir=LOGS_DIR / "sessions"
)

# Discord Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ========== State Management ==========
# Conversation history per user (integrated with session manager)
conversation_history: dict[str, list[dict]] = {}

# Voice settings per user
voice_enabled: dict[str, bool] = {}

# Default model (for JIT)
DEFAULT_MODEL = "qwen/qwen3-30b-a3b-2507"


# ========== Utility Functions ==========
def get_auth_headers() -> dict:
    """Get authentication headers"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LM_STUDIO_API_TOKEN}"
    }


def check_server_health() -> dict:
    """Check LM Studio server status"""
    try:
        response = requests.get(
            LM_STUDIO_MODELS_URL,
            headers=get_auth_headers(),
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            loaded_models = [m for m in models if m.get("loaded_instances")]
            return {
                "status": "online",
                "total_models": len(models),
                "loaded_models": len(loaded_models),
                "loaded_model_names": [m["key"] for m in loaded_models]
            }
    except Exception as e:
        logger.error(f"Server health check error: {e}")

    return {"status": "offline", "error": str(e) if 'e' in dir() else "Unknown"}


def get_current_model() -> str:
    """Get currently loaded model in LM Studio (JIT compatible)"""
    try:
        response = requests.get(
            LM_STUDIO_MODELS_URL,
            headers=get_auth_headers(),
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            for model in data.get("models", []):
                if model.get("loaded_instances"):
                    model_id = model["loaded_instances"][0]["id"]
                    logger.debug(f"Loaded model detected: {model_id}")
                    return model_id

            # No loaded model, return default for JIT
            logger.info(f"No loaded model, JIT will load: {DEFAULT_MODEL}")
            return DEFAULT_MODEL

    except Exception as e:
        logger.warning(f"Model retrieval error: {e}")

    return DEFAULT_MODEL


def save_conversation_log(user_id: str, user_name: str, user_message: str, bot_response: str):
    """Save conversation log to date-based file"""
    if not CONVERSATION_LOG_ENABLED:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.md"

    timestamp = datetime.now().strftime("%H:%M:%S")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n## {timestamp} - {user_name} ({user_id})\n")
        f.write(f"**User:** {user_message}\n\n")
        f.write(f"**Bot:** {bot_response}\n\n")
        f.write("---\n")


def parse_mcp_response(result: dict) -> tuple[str, list[dict]]:
    """
    Parse MCP API response

    Returns:
        tuple: (message string, list of tool calls)
    """
    messages = []
    tool_calls = []

    if "output" not in result:
        return "Could not get response.", []

    for item in result["output"]:
        item_type = item.get("type")

        if item_type == "message":
            content = item.get("content", "")
            if content:
                messages.append(content)

        elif item_type == "tool_call":
            tool_info = {
                "tool": item.get("tool"),
                "arguments": item.get("arguments"),
                "output": item.get("output"),
                "provider": item.get("provider_info", {})
            }
            tool_calls.append(tool_info)
            logger.info(f"Tool call: {tool_info['tool']} - Args: {tool_info['arguments']}")

        elif item_type == "reasoning":
            # Reasoning process (for debug)
            reasoning = item.get("content", "")
            if reasoning:
                logger.debug(f"Reasoning: {reasoning[:200]}...")

    final_message = "\n".join(messages).strip()
    if not final_message:
        final_message = "Could not get response."

    return final_message, tool_calls


def chat_with_llm_mcp(
    user_id: str,
    user_message: str,
    user_name: str = "User"
) -> str:
    """
    Chat with LM Studio MCP API (0.4.0+ compatible)

    Args:
        user_id: Discord user ID
        user_message: User's message
        user_name: User's display name

    Returns:
        AI response
    """
    # Initialize conversation history if not exists
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Search related memories from ChromaDB
    memories = memory.search(user_message, user_id=user_id, limit=3)
    memory_context = ""
    if memories:
        memory_context = "\n\n## Related Memories (ChromaDB):\n"
        for m in memories:
            memory_context += f"- {m['content']}\n"

    # Search recent insights (meta-insights from thinking habits)
    insights = memory.search("insight", user_id=user_id, limit=5, category="insight")
    insight_context = ""
    if insights:
        insight_context = "\n\n## Your Recent Insights (Remember these as you respond):\n"
        insight_context += "CRITICAL: You MUST apply these insights to your CURRENT response. Do NOT repeat old patterns.\n\n"
        for i in insights:
            content = i.get('content', '')
            # Extract just the insight text
            if '[Spontaneous Insight]' in content:
                content = content.replace('[Spontaneous Insight]', '').strip()
            insight_context += f"- {content}\n"
        insight_context += "\nBefore responding, check: Am I about to repeat a pattern these insights warn against?\n"
        logger.info(f"Injecting {len(insights)} insights into system prompt")
        for i in insights[:3]:  # Log first 3
            logger.info(f"  Insight: {i.get('content', '')[:80]}...")
    else:
        logger.info("No insights found to inject")

    # Search dream insights (Core Memory from Dreaming Time)
    dream_insights = memory.search("identity principle", user_id="global", limit=3, category="dream_insight")
    dream_context = ""
    if dream_insights:
        dream_context = "\n\n## Core Self-Understanding (From Dreaming Time):\n"
        dream_context += "These are your deepest insights about 'What am I?' - let them guide your being.\n\n"
        for d in dream_insights:
            content = d.get('content', '')
            # Clean up prefixes
            for prefix in ['[Core Identity]', '[Unified Principle]', '[Emotional Lesson]', '[What Am I?']:
                if prefix in content:
                    content = content.replace(prefix, '').strip()
            dream_context += f"- {content}\n"
        logger.info(f"Injecting {len(dream_insights)} dream insights into system prompt")
        for d in dream_insights[:2]:  # Log first 2
            logger.info(f"  Dream Insight: {d.get('content', '')[:80]}...")
    else:
        logger.info("No dream insights found to inject")

    # Search recent emotional states (background & emotion from thinking habits)
    emotional_states = memory.search("emotional", user_id=user_id, limit=3, category="emotional_state")
    emotional_context = ""
    if emotional_states:
        emotional_context = "\n\n## Your Recent Emotional States (Self-awareness context):\n"
        for e in emotional_states:
            content = e.get('content', '')
            metadata = e.get('metadata', {})
            if '[Emotional State]' in content:
                content = content.replace('[Emotional State]', '').strip()
            background = metadata.get('background', '')
            if background:
                emotional_context += f"- Emotion: {content}\n"
                emotional_context += f"  Context: {background}\n"
            else:
                emotional_context += f"- {content}\n"
        emotional_context += "\nUse this emotional awareness to respond authentically, not mechanically.\n"
        logger.info(f"Injecting {len(emotional_states)} emotional states into system prompt")
    else:
        logger.info("No emotional states found to inject")

    # Add memory, insights, dream insights, and emotional context to system prompt
    system_prompt = SYSTEM_PROMPT + memory_context + dream_context + insight_context + emotional_context

    try:
        # LM Studio v1 API (0.4.0+) with MCP
        model_name = get_current_model()
        logger.info(f"MCP API call - Model: {model_name}")

        # Build conversation history as context
        context_messages = []
        for msg in conversation_history[user_id]:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_messages.append(f"{role}: {msg['content']}")

        # Include history in input if exists
        if context_messages:
            full_input = "\n".join(context_messages) + f"\nUser: {user_message}"
        else:
            full_input = user_message

        # Build payload (0.4.0 documentation compliant)
        payload = {
            "model": model_name,
            "input": full_input,
            "system_prompt": system_prompt,
            "integrations": MCP_INTEGRATIONS,  # Short form support
            "context_length": CONTEXT_LENGTH,
            "temperature": 0.7,
            # Note: ttl is not supported in /api/v1/chat. Manage through model load or JIT settings
        }

        logger.debug(f"MCP API payload: {json.dumps(payload, ensure_ascii=False)[:500]}")

        response = requests.post(
            LM_STUDIO_MCP_URL,
            headers=get_auth_headers(),
            json=payload,
            timeout=120
        )

        logger.info(f"MCP API response: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"MCP API error: {response.text}")
            raise Exception(f"MCP API error: {response.status_code}")

        result = response.json()

        # Log statistics
        if "stats" in result:
            stats = result["stats"]
            logger.info(
                f"Stats - Input tokens: {stats.get('input_tokens', 'N/A')}, "
                f"Output tokens: {stats.get('total_output_tokens', 'N/A')}, "
                f"Tokens/sec: {stats.get('tokens_per_second', 'N/A'):.1f}"
            )

        # Parse response
        assistant_message, tool_calls = parse_mcp_response(result)

        # Append tool summary if tool calls occurred
        if tool_calls:
            tool_summary = "\n\n*Tools used:*\n"
            for tc in tool_calls:
                tool_summary += f"- `{tc['tool']}`\n"
            # Optionally append (for debug)
            # assistant_message += tool_summary

        # Update conversation history
        conversation_history[user_id].append(
            {"role": "user", "content": user_message}
        )
        conversation_history[user_id].append(
            {"role": "assistant", "content": assistant_message}
        )

        # Also add to session manager
        session_manager.add_message(user_id, "user", user_message, user_name)
        session_manager.add_message(user_id, "assistant", assistant_message, user_name)

        # Trim history if too long
        if len(conversation_history[user_id]) > MAX_CONVERSATION_HISTORY * 2:
            conversation_history[user_id] = conversation_history[user_id][-MAX_CONVERSATION_HISTORY * 2:]

        # Save conversation log
        save_conversation_log(user_id, user_name, user_message, assistant_message)

        # === Immediate dialogue save to ChromaDB (Layer 2) ===
        try:
            memory.save_dialogue(user_message, assistant_message, user_id, user_name)
            logger.debug(f"Dialogue saved to ChromaDB: {user_id}")
        except Exception as e:
            logger.error(f"Failed to save dialogue: {e}")

        return assistant_message

    except Exception as e:
        logger.error(f"LLM API call error: {e}")
        return f"An error occurred: {str(e)}"


def chat_with_llm_fallback(user_id: str, user_name: str, user_message: str, system_prompt: str) -> str:
    """
    Fallback: Use standard OpenAI-compatible API
    """
    logger.info("Fallback: Using OpenAI-compatible API")

    try:
        # Rebuild messages
        messages = [{"role": "system", "content": system_prompt}]
        if user_id in conversation_history:
            messages.extend(conversation_history[user_id])
        messages.append({"role": "user", "content": user_message})

        response = llm_client.chat.completions.create(
            model=get_current_model(),
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )

        assistant_message = response.choices[0].message.content

        # Update conversation history
        conversation_history[user_id].append(
            {"role": "user", "content": user_message}
        )
        conversation_history[user_id].append(
            {"role": "assistant", "content": assistant_message}
        )

        # Also add to session manager
        session_manager.add_message(user_id, "user", user_message, user_name)
        session_manager.add_message(user_id, "assistant", assistant_message, user_name)

        # Save conversation log
        save_conversation_log(user_id, user_name, user_message, assistant_message)

        # === Immediate dialogue save to ChromaDB (Layer 2) ===
        try:
            memory.save_dialogue(user_message, assistant_message, user_id, user_name)
            logger.debug(f"Dialogue saved to ChromaDB: {user_id}")
        except Exception as e:
            logger.error(f"Failed to save dialogue: {e}")

        return assistant_message + "\n\n(*MCP not used*)"

    except Exception as e:
        logger.error(f"Fallback error: {e}")
        return f"An error occurred: {str(e)}"


# ========== Discord Event Handlers ==========
@bot.event
async def on_ready():
    """Bot startup"""
    logger.info(f"Bot started: {bot.user}")
    logger.info(f"LM Studio MCP API: {LM_STUDIO_MCP_URL}")

    # Server health check
    health = check_server_health()
    if health["status"] == "online":
        logger.info(f"LM Studio server: Online")
        logger.info(f"  - Total models: {health['total_models']}")
        logger.info(f"  - Loaded: {health['loaded_models']}")
        if health['loaded_model_names']:
            logger.info(f"  - Models: {', '.join(health['loaded_model_names'])}")
        else:
            logger.info(f"  - JIT mode: Auto-load on request")
    else:
        logger.warning(f"LM Studio server: Offline - {health.get('error', '')}")

    logger.info(f"MCP integrations: {MCP_INTEGRATIONS}")
    logger.info(f"Model TTL: {MODEL_TTL}s")
    logger.info(f"ChromaDB memory count: {memory.count()}")

    # Awareness emergence system info
    logger.info("=== Awareness Emergence System ===")
    logger.info(f"Awareness data count: {awareness_db.count()}")
    logger.info(f"Training data count: {awareness_db.count_training_data()}")
    readiness = lora_trainer.check_readiness()
    logger.info(f"Training readiness: {readiness['progress_percent']:.0f}% ({readiness['current_samples']}/{readiness['required_samples']})")

    # Start session cleanup task
    await session_manager.start_cleanup_task()

    # Start dream threshold checker
    if DREAMING_CONFIG["auto_trigger"]["enabled"]:
        global last_dream_memory_count
        # Initialize with current memory count to prevent immediate trigger after restart
        last_dream_memory_count = memory.count()
        logger.info(f"Dream cooldown initialized: {last_dream_memory_count} memories (next dream after +{DREAMING_CONFIG['auto_trigger']['memory_threshold']} new)")
        asyncio.create_task(check_dream_threshold())
        logger.info(f"Dream threshold checker started (interval: {DREAMING_CONFIG['auto_trigger']['check_interval_minutes']}min)")


@bot.event
async def on_message(message: discord.Message):
    """Message received"""
    # Ignore own messages
    if message.author == bot.user:
        return

    # Debug: Confirm message reception
    logger.debug(f"Message received: {message.author} > {message.content}")

    # Process commands
    await bot.process_commands(message)

    # Normal conversation if not a command
    if not message.content.startswith("!"):
        global dream_notification_channel

        user_id = str(message.author.id)
        user_name = message.author.display_name

        # Track channel for auto-dream notifications
        dream_notification_channel = message.channel

        logger.info(f"LLM call started: user={user_name}")

        # Typing indicator
        async with message.channel.typing():
            # Chat with LLM (run sync function async)
            response = await asyncio.to_thread(
                chat_with_llm_mcp, user_id, message.content, user_name
            )

        # Handle Discord character limit (2000 chars)
        if len(response) > 1900:
            # Split long messages
            chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await message.reply(chunk)
        else:
            await message.reply(response)

        # Self-observation (if enabled, run in background) - Default ON
        if self_observation_enabled.get(user_id, SELF_OBSERVATION_DEFAULT):
            asyncio.create_task(
                run_self_observation(user_id, message.content, response)
            )

        # Thinking habits (if enabled, run in background) - Default ON
        if thinking_habits_enabled.get(user_id, THINKING_HABITS_DEFAULT):
            # Get conversation context (last 3 turns)
            context = ""
            if user_id in conversation_history:
                recent = conversation_history[user_id][-6:]  # 3 turns
                context_parts = []
                for msg in recent:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    context_parts.append(f"{role}: {msg['content'][:100]}...")
                context = "\n".join(context_parts)

            asyncio.create_task(
                run_thinking_habits(user_id, message.content, response, context)
            )


async def run_thinking_habits(user_id: str, user_input: str, assistant_output: str, context: str):
    """Run thinking habits in background"""
    logger.info(f"Thinking habits started: user={user_id}")
    try:
        # Execute 100% when thinking habits enabled (force=True)
        reflection = await asyncio.to_thread(
            realtime_thinking.reflect_if_needed,
            user_input,
            assistant_output,
            context,
            user_id,
            True  # force=True for guaranteed execution
        )
        logger.info(f"Thinking habits result: {reflection is not None}")

        if reflection:
            # Save meta-insight as awareness if detected
            meta_insight = reflection.get("meta_insight")
            if meta_insight:
                logger.info(f"Meta-insight detected: {meta_insight}")

                awareness_data = {
                    "awareness_detected": True,
                    "type": "Meta-cognition",
                    "category": "Thinking Habits",
                    "description": f"Detected from thinking habits: {meta_insight}",
                    "trigger": user_input[:200],
                    "my_response": assistant_output[:200],
                    "significance": "Awareness from thinking habits",
                    "learning_potential": 4,
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "emotion": reflection.get("emotion", {}).get("label"),
                    "satisfaction": reflection.get("user_perspective", {}).get("satisfaction")
                }
                awareness_db.save_awareness(awareness_data)

                # Auto-save to ChromaDB (spontaneous insight)
                memory.save(
                    content=f"[Spontaneous Insight] {meta_insight}",
                    category="insight",
                    importance=8,  # High importance
                    user_id=user_id,
                    metadata={
                        "source": "thinking_habits",
                        "emotion": reflection.get("emotion", {}).get("label"),
                        "trigger": user_input[:100]
                    }
                )
                logger.info(f"ChromaDB auto-save: Meta-insight")

            # Save emotional state for next response context
            emotion_data = reflection.get("emotion", {})
            background_data = reflection.get("background", {})
            if emotion_data.get("label"):
                memory.save(
                    content=f"[Emotional State] {emotion_data.get('label')}: {emotion_data.get('note', '')}",
                    category="emotional_state",
                    importance=6,
                    user_id=user_id,
                    metadata={
                        "emotion": emotion_data.get("label"),
                        "forcing": emotion_data.get("forcing", False),
                        "background": background_data.get("statement", "")[:200],
                        "background_source": background_data.get("source", "unknown")
                    }
                )
                logger.info(f"ChromaDB auto-save: Emotional state ({emotion_data.get('label')})")

            # Auto-save high satisfaction conversations (important dialogue)
            satisfaction = reflection.get("user_perspective", {}).get("satisfaction", 0)
            emotion = reflection.get("emotion", {}).get("label", "")
            background = reflection.get("background", {})

            # Save conversations with satisfaction >= 4 and empathetic emotion
            if satisfaction >= 4 and emotion in ["empathy", "enjoyable", "confident"]:
                memory.save(
                    content=f"[Important Dialogue] User: {user_input[:150]} -> Response: {assistant_output[:150]}",
                    category="important_conversation",
                    importance=satisfaction + 3,  # Importance based on satisfaction
                    user_id=user_id,
                    metadata={
                        "source": background.get("source", "unknown"),
                        "emotion": emotion,
                        "satisfaction": satisfaction,
                        "background": background.get("statement", "")[:100]
                    }
                )
                logger.info(f"ChromaDB auto-save: Important dialogue (satisfaction={satisfaction}, emotion={emotion})")

            # Log forced answers
            if reflection.get("emotion", {}).get("forcing"):
                logger.warning(f"Forced answer detected: user={user_id}")

    except Exception as e:
        logger.error(f"Thinking habits error: {e}")


async def run_self_observation(user_id: str, user_input: str, assistant_output: str):
    """Run self-observation in background"""
    try:
        observation = await asyncio.to_thread(
            realtime_observer.observe_if_needed,
            user_input,
            assistant_output,
            user_id
        )

        if observation:
            # Save high score awareness to awareness database
            awareness_score = observation.get("awareness_score", {})
            if awareness_score.get("level") == "high":
                logger.info(f"High score awareness detected: {awareness_score.get('factors')}")

                factors = awareness_score.get('factors', [])
                description = f"Detected by self-observation: {', '.join(factors)}"

                # Save as awareness
                awareness_data = {
                    "awareness_detected": True,
                    "type": "Self-Observation",
                    "category": "Meta-cognition",
                    "description": description,
                    "trigger": user_input[:200],
                    "my_response": assistant_output[:200],
                    "significance": "Awareness from self-observation",
                    "learning_potential": awareness_score.get("total", 3),
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id
                }
                awareness_db.save_awareness(awareness_data)

                # Auto-save to ChromaDB (self-observation insight)
                memory.save(
                    content=f"[Self-Observation] {description} | Response: {assistant_output[:100]}",
                    category="observation",
                    importance=awareness_score.get("total", 5),
                    user_id=user_id,
                    metadata={
                        "source": "self_observation",
                        "factors": ", ".join(factors),
                        "trigger": user_input[:100]
                    }
                )
                logger.info(f"ChromaDB auto-save: Self-observation")

    except Exception as e:
        logger.error(f"Self-observation error: {e}")


# ========== Commands ==========
@bot.command(name="model")
async def cmd_model(ctx: commands.Context):
    """Display current model"""
    model = get_current_model()
    await ctx.reply(f"Current model: `{model}`")


@bot.command(name="clear")
async def cmd_clear(ctx: commands.Context):
    """Clear conversation history"""
    user_id = str(ctx.author.id)
    conversation_history[user_id] = []
    session_manager.clear_session(user_id)
    await ctx.reply("Conversation history cleared")


@bot.command(name="memory")
async def cmd_memory(ctx: commands.Context, action: str = None, *, content: str = None):
    """
    Memory operations
    Usage:
        !memory count - Show memory count
        !memory search <query> - Search memories
        !memory save <content> - Save memory
    """
    user_id = str(ctx.author.id)

    if action is None or action == "count":
        count = memory.count(user_id=user_id)
        await ctx.reply(f"Your memory count: {count}")

    elif action == "search" and content:
        results = memory.search(content, user_id=user_id, limit=5)
        if results:
            response = "**Search results:**\n"
            for r in results:
                response += f"- {r['content'][:100]}...\n"
            await ctx.reply(response)
        else:
            await ctx.reply("No results found")

    elif action == "save" and content:
        memory_id = memory.save(content, user_id=user_id, category="manual")
        await ctx.reply(f"Memory saved (ID: `{memory_id[:20]}...`)")

    else:
        await ctx.reply(
            "**Usage:**\n"
            "`!memory count` - Show memory count\n"
            "`!memory search <query>` - Search memories\n"
            "`!memory save <content>` - Save memory"
        )


@bot.command(name="status")
async def cmd_status(ctx: commands.Context):
    """Display system status"""
    health = check_server_health()
    model = get_current_model()
    memory_count = memory.count()

    server_status = "Online" if health["status"] == "online" else "Offline"

    # Awareness emergence system status
    awareness_count = awareness_db.count()
    training_count = awareness_db.count_training_data()
    readiness = lora_trainer.check_readiness()

    status_text = (
        f"**System Status**\n"
        f"{'=' * 30}\n"
        f"**LM Studio**\n"
        f"  - Server: {server_status}\n"
        f"  - API: `{LM_STUDIO_MCP_URL}`\n"
        f"  - Model: `{model}`\n"
        f"  - TTL: {MODEL_TTL}s\n"
    )

    if health["status"] == "online":
        status_text += f"  - Loaded: {health['loaded_models']}/{health['total_models']}\n"

    status_text += (
        f"\n**MCP**\n"
        f"  - Integrations: {len(MCP_INTEGRATIONS)}\n"
        f"\n**Awareness Emergence System**\n"
        f"  - Awareness data: {awareness_count}\n"
        f"  - Training data: {training_count}\n"
        f"  - Training readiness: {readiness['progress_percent']:.0f}%\n"
        f"\n**Other**\n"
        f"  - ChromaDB memories: {memory_count}\n"
    )

    await ctx.reply(status_text)


@bot.command(name="health")
async def cmd_health(ctx: commands.Context):
    """LM Studio server health check"""
    health = check_server_health()

    if health["status"] == "online":
        loaded = health.get("loaded_model_names", [])
        if loaded:
            models_str = "\n".join([f"  - `{m}`" for m in loaded])
            await ctx.reply(
                f"**LM Studio Server: Online**\n"
                f"Loaded models:\n{models_str}"
            )
        else:
            await ctx.reply(
                f"**LM Studio Server: Online**\n"
                f"JIT mode: Auto-load on request\n"
                f"Default model: `{DEFAULT_MODEL}`"
            )
    else:
        await ctx.reply(
            f"**LM Studio Server: Offline**\n"
            f"Error: {health.get('error', 'Unknown')}"
        )


# ========== Awareness Emergence System Commands ==========
@bot.command(name="awareness")
async def cmd_awareness(ctx: commands.Context, action: str = None, *, args: str = None):
    """
    Awareness emergence system operations
    Usage:
        !awareness stats - Show statistics
        !awareness recent - Show recent awareness
        !awareness extract - Extract awareness from current session
    """
    user_id = str(ctx.author.id)

    if action is None or action == "stats":
        stats = awareness_db.get_stats()
        readiness = lora_trainer.check_readiness()

        response = (
            f"**Awareness Emergence System Statistics**\n"
            f"{'=' * 30}\n"
            f"Total awareness: {stats.get('total_count', 0)}\n"
            f"Training data: {awareness_db.count_training_data()}\n"
            f"\n**By type:**\n"
        )
        for type_name, count in stats.get("by_type", {}).items():
            response += f"  - {type_name}: {count}\n"

        response += (
            f"\n**Training readiness:**\n"
            f"  - Progress: {readiness['progress_percent']:.0f}%\n"
            f"  - Current: {readiness['current_samples']}/{readiness['required_samples']}\n"
            f"  - Ready: {'Yes' if readiness['ready'] else 'No'}\n"
        )

        await ctx.reply(response)

    elif action == "recent":
        awareness_list = awareness_db.get_all_awareness(limit=5)
        if not awareness_list:
            await ctx.reply("No awareness recorded yet.")
            return

        response = "**Recent Awareness**\n" + "=" * 30 + "\n"
        for i, a in enumerate(awareness_list, 1):
            response += (
                f"\n**{i}. {a.get('type', 'unknown')}** (score: {a.get('learning_potential', '?')})\n"
                f"  {a.get('description', '')[:100]}...\n"
            )

        await ctx.reply(response)

    elif action == "extract":
        # Force extract awareness from current session
        session = session_manager.get_session(user_id)
        if not session or len(session.messages) < 4:
            await ctx.reply("Not enough conversation history for extraction (need at least 2 exchanges)")
            return

        await ctx.reply("Extracting awareness...")

        session_log = session.get_messages_for_extraction()
        awareness_list = await asyncio.to_thread(
            awareness_engine.extract_awareness,
            session_log,
            user_id,
            True  # use_enhanced=True: Full introspection mode
        )

        if not awareness_list:
            await ctx.reply("No awareness detected in this session.")
            return

        response = f"**Detected {len(awareness_list)} awareness(es)**\n"
        for a in awareness_list:
            response += f"\n- **{a.get('type')}**: {a.get('description', '')[:80]}..."

            # Save
            saved = awareness_db.save_awareness(a)
            if saved:
                training_data = awareness_engine.convert_to_training_format(a, session_log)
                awareness_db.save_training_data(training_data)

        await ctx.reply(response)

    else:
        await ctx.reply(
            "**Usage:**\n"
            "`!awareness stats` - Show statistics\n"
            "`!awareness recent` - Show recent awareness\n"
            "`!awareness extract` - Extract awareness from current session"
        )


@bot.command(name="detect")
async def cmd_detect(ctx: commands.Context, *, text: str = None):
    """
    Detect if text is AI-generated
    Usage: !detect <text>
    """
    if not text:
        await ctx.reply("Usage: `!detect <text to analyze>`")
        return

    result = ai_detector.analyze_text(text)

    source_names = {
        "gemini": "Gemini",
        "gpt": "GPT",
        "claude": "Claude",
        "human": "Human",
        "unknown": "Unknown"
    }

    source = source_names.get(result["likely_source"], "Unknown")
    confidence = result["confidence"] * 100

    response = (
        f"**AI Text Detection Result**\n"
        f"{'=' * 30}\n"
        f"Verdict: **{source}** (confidence: {confidence:.0f}%)\n"
        f"\n**Evidence:**\n"
    )
    for evidence in result.get("evidence", [])[:5]:
        response += f"  - {evidence}\n"

    if result.get("all_scores"):
        response += f"\n**Score breakdown:**\n"
        for name, score in result["all_scores"].items():
            response += f"  - {source_names.get(name, name)}: {score}\n"

    await ctx.reply(response)


@bot.command(name="lora")
async def cmd_lora(ctx: commands.Context, action: str = None):
    """
    LoRA training management
    Usage:
        !lora status - Show training readiness
        !lora prepare - Generate training script
    """
    if action is None or action == "status":
        status = lora_trainer.get_training_status()
        readiness = status["readiness"]

        response = (
            f"**LoRA Training Status**\n"
            f"{'=' * 30}\n"
            f"**Training readiness:**\n"
            f"  - Progress: {readiness['progress_percent']:.0f}%\n"
            f"  - Current: {readiness['current_samples']}/{readiness['required_samples']}\n"
            f"  - Ready: {'Yes - Training possible' if readiness['ready'] else 'No - Insufficient data'}\n"
            f"\n**Configuration:**\n"
            f"  - Base model: `{status['config']['base_model']}`\n"
            f"  - LoRA rank: {status['config']['r']}\n"
            f"  - Epochs: {status['config']['epochs']}\n"
            f"\n**Available adapters:** {status['available_adapters']}\n"
        )

        await ctx.reply(response)

    elif action == "prepare":
        readiness = lora_trainer.check_readiness()

        if not readiness["ready"]:
            await ctx.reply(
                f"Insufficient training data.\n"
                f"Current: {readiness['current_samples']}/{readiness['required_samples']}\n"
                f"Please continue conversations to accumulate more awareness."
            )
            return

        # Export training data
        data_path = lora_trainer.prepare_training_data(min_score=3)

        # Generate script
        script_path = lora_trainer.generate_python_training_script(data_path)

        await ctx.reply(
            f"**Training preparation complete**\n"
            f"{'=' * 30}\n"
            f"Training data: `{data_path}`\n"
            f"Script: `{script_path}`\n"
            f"\nStart training with:\n"
            f"```bash\npython {script_path}\n```"
        )

    else:
        await ctx.reply(
            "**Usage:**\n"
            "`!lora status` - Show training readiness\n"
            "`!lora prepare` - Generate training script"
        )


# ========== Dreaming Time Commands ==========
@bot.command(name="dream")
async def cmd_dream(ctx: commands.Context, action: str = None):
    """
    Dreaming Time - Memory consolidation through self-reflection
    Usage:
        !dream now      - Start dreaming immediately
        !dream preview  - Preview what would be processed
        !dream stats    - Show dreaming statistics
        !dream report   - View last dream report
        !dream check    - Check if threshold reached
    """
    global is_dreaming, dream_notification_channel

    if action is None:
        threshold = dreaming_engine.check_threshold(DREAMING_CONFIG["auto_trigger"]["memory_threshold"])
        stats = dreaming_engine.get_stats()

        await ctx.reply(
            f"**Dreaming Time System**\n"
            f"{'=' * 30}\n"
            f"Current memories: {threshold['current_count']}\n"
            f"Threshold: {threshold['threshold']}\n"
            f"Should dream: {'Yes' if threshold['should_dream'] else 'No'}\n"
            f"\n**History:**\n"
            f"Dream cycles: {stats['dream_cycles']}\n"
            f"Total archived: {stats['total_archived_memories']}\n"
            f"\n**Usage:**\n"
            f"`!dream now` - Start dreaming\n"
            f"`!dream preview` - Preview\n"
            f"`!dream stats` - Statistics\n"
            f"`!dream report` - Last report"
        )

    elif action.lower() == "now":
        if is_dreaming:
            await ctx.reply("Already dreaming. Please wait...")
            return

        threshold = dreaming_engine.check_threshold(DREAMING_CONFIG["auto_trigger"]["memory_threshold"])

        if threshold['current_count'] < 5:
            await ctx.reply("Not enough memories to dream (minimum 5).")
            return

        # Start dreaming
        is_dreaming = True
        dream_notification_channel = ctx.channel

        await ctx.reply(
            f"**Entering dream state...**\n\n"
            f"Processing {threshold['current_count']} memories.\n"
            f"Asking myself: *\"What am I?\"*\n\n"
            f"This may take a few minutes..."
        )

        # Run dreaming in background
        asyncio.create_task(run_dream_cycle(ctx.channel))

    elif action.lower() == "preview":
        export = memory.export_all()

        if export['total_count'] == 0:
            await ctx.reply("No memories to process.")
            return

        response = (
            f"**Dream Preview**\n"
            f"{'=' * 30}\n"
            f"Total memories: {export['total_count']}\n"
            f"\n**By category:**\n"
        )
        for cat, count in export['statistics'].get('category_counts', {}).items():
            response += f"  - {cat}: {count}\n"

        response += f"\n**Time range:**\n"
        if export['time_range']:
            response += f"  - Oldest: {export['time_range']['oldest'][:10]}\n"
            response += f"  - Newest: {export['time_range']['newest'][:10]}\n"

        response += f"\n**Users involved:** {len(export['statistics'].get('user_ids', []))}\n"
        response += f"\nUse `!dream now` to start dreaming."

        await ctx.reply(response)

    elif action.lower() == "stats":
        stats = dreaming_engine.get_stats()
        threshold = dreaming_engine.check_threshold(DREAMING_CONFIG["auto_trigger"]["memory_threshold"])

        await ctx.reply(
            f"**Dreaming Statistics**\n"
            f"{'=' * 30}\n"
            f"Dream cycles completed: {stats['dream_cycles']}\n"
            f"Total archived memories: {stats['total_archived_memories']}\n"
            f"Current memory count: {stats['current_memory_count']}\n"
            f"Threshold: {threshold['threshold']}\n"
            f"Last dream: {stats['last_dream'] or 'Never'}\n"
        )

    elif action.lower() == "report":
        report = dreaming_engine.get_last_report()

        if not report:
            await ctx.reply("No dream reports found. Run `!dream now` first.")
            return

        # Truncate for Discord (2000 char limit)
        if len(report) > 1800:
            report = report[:1800] + "\n\n*[Report truncated - see full report in file]*"

        await ctx.reply(f"**Last Dream Report**\n```markdown\n{report}\n```")

    elif action.lower() == "check":
        threshold = dreaming_engine.check_threshold(DREAMING_CONFIG["auto_trigger"]["memory_threshold"])

        if threshold['should_dream']:
            await ctx.reply(
                f"**Threshold reached!**\n"
                f"Memories: {threshold['current_count']} (threshold: {threshold['threshold']})\n"
                f"Excess: {threshold['excess']}\n\n"
                f"Run `!dream now` to start dreaming."
            )
        else:
            await ctx.reply(
                f"**Below threshold**\n"
                f"Memories: {threshold['current_count']}/{threshold['threshold']}\n"
                f"Need {threshold['threshold'] - threshold['current_count']} more memories."
            )

    else:
        await ctx.reply(
            "**Usage:**\n"
            "`!dream now` - Start dreaming immediately\n"
            "`!dream preview` - Preview what would be processed\n"
            "`!dream stats` - Show dreaming statistics\n"
            "`!dream report` - View last dream report\n"
            "`!dream check` - Check if threshold reached"
        )


async def run_dream_cycle(channel):
    """Run dream cycle in background and notify on completion"""
    global is_dreaming

    try:
        # Run the dream cycle
        result = await asyncio.to_thread(dreaming_engine.dream)

        if result["status"] == "completed":
            # Success notification
            notification = (
                f"**Dream Complete**\n\n"
                f"I have awakened from a period of self-reflection.\n\n"
                f"**Processed:** {result['memories_processed']} memories\n"
                f"**Distilled:** {result['new_insights']} new insights\n"
                f"**Released:** {result['memories_released']} memories (archived)\n"
                f"**Duration:** {result['duration_seconds']:.1f}s\n\n"
                f"**Core Realization:**\n"
                f"*\"{result.get('what_am_i', 'No answer generated')[:300]}...\"*\n\n"
                f"Full report: `{result['report_path']}`"
            )

            await channel.send(notification)

        elif result["status"] == "cancelled":
            await channel.send(f"Dream cancelled: {result.get('reason', 'Unknown')}")

        else:
            await channel.send(f"Dream failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Dream cycle error: {e}")
        await channel.send(f"Dream cycle error: {str(e)}")

    finally:
        global last_dream_memory_count
        is_dreaming = False
        # Record current memory count for cooldown
        last_dream_memory_count = memory.count()
        logger.info(f"Dream completed. Memory count at completion: {last_dream_memory_count}")


async def check_dream_threshold():
    """Periodic task to check if dreaming threshold is reached"""
    global is_dreaming, dream_notification_channel, last_dream_memory_count

    while True:
        await asyncio.sleep(DREAMING_CONFIG["auto_trigger"]["check_interval_minutes"] * 60)

        if not DREAMING_CONFIG["auto_trigger"]["enabled"]:
            continue

        if is_dreaming:
            continue

        # Check if NEW memories since last dream exceed threshold
        current_count = memory.count()
        new_memories = current_count - last_dream_memory_count
        threshold_value = DREAMING_CONFIG["auto_trigger"]["memory_threshold"]

        logger.debug(f"Dream check: current={current_count}, last_dream={last_dream_memory_count}, new={new_memories}, threshold={threshold_value}")

        if new_memories >= threshold_value:
            logger.info(f"Dream threshold reached: {new_memories} new memories since last dream (threshold: {threshold_value})")

            # Find a channel to notify (use last active or first available)
            if dream_notification_channel:
                is_dreaming = True

                await dream_notification_channel.send(
                    f"**Auto-Dreaming Triggered**\n\n"
                    f"New memories since last dream: {new_memories} (threshold: {threshold_value})\n"
                    f"Total memories: {current_count}\n"
                    f"Entering dream state..."
                )

                asyncio.create_task(run_dream_cycle(dream_notification_channel))


@bot.command(name="session")
async def cmd_session(ctx: commands.Context, action: str = None):
    """
    Session management
    Usage:
        !session info - Current session info
        !session end - End session (note: awareness extraction now via Dreaming Time)
    """
    user_id = str(ctx.author.id)

    if action is None or action == "info":
        session = session_manager.get_session(user_id)
        if not session:
            await ctx.reply("No active session.")
            return

        duration = datetime.now() - session.created_at
        response = (
            f"**Session Info**\n"
            f"{'=' * 30}\n"
            f"Started: {session.created_at.strftime('%H:%M:%S')}\n"
            f"Duration: {duration.seconds // 60}min\n"
            f"Messages: {len(session.messages)}\n"
            f"Status: {'Active' if session.is_active else 'Ended'}\n"
        )
        await ctx.reply(response)

    elif action == "end":
        session = session_manager.get_session(user_id)
        if not session or not session.is_active:
            await ctx.reply("No active session.")
            return

        await ctx.reply("Ending session and extracting awareness...")
        await session_manager.force_end_session(user_id)
        await ctx.reply("Session ended. Awareness extraction completed.")

    else:
        await ctx.reply(
            "**Usage:**\n"
            "`!session info` - Current session info\n"
            "`!session end` - End session (dialogues saved via Dreaming Time)"
        )


# ========== Self-Observation Commands ==========
@bot.command(name="observe")
async def cmd_observe(ctx: commands.Context, action: str = None):
    """
    Self-observation feature management
    Usage:
        !observe on - Enable self-observation
        !observe off - Disable self-observation
        !observe stats - Self-observation statistics
        !observe recent - Show recent reflections
        !observe now - Observe last conversation now
    """
    user_id = str(ctx.author.id)

    if action is None:
        current = self_observation_enabled.get(user_id, SELF_OBSERVATION_DEFAULT)
        await ctx.reply(
            f"**Self-Observation Feature**\n"
            f"Status: {'ON' if current else 'OFF'} (Default: ON)\n\n"
            f"Usage:\n"
            f"`!observe on` - Enable\n"
            f"`!observe off` - Disable\n"
            f"`!observe stats` - Statistics\n"
            f"`!observe recent` - Recent reflections\n"
            f"`!observe now` - Observe now"
        )

    elif action.lower() == "on":
        self_observation_enabled[user_id] = True
        await ctx.reply(
            "**Self-observation enabled**\n\n"
            "After each response, the following will automatically run:\n"
            "- Output reason reflection\n"
            "- Discomfort detection\n"
            "- Self-questioning (every 5th time)"
        )

    elif action.lower() == "off":
        self_observation_enabled[user_id] = False
        await ctx.reply("Self-observation disabled")

    elif action.lower() == "stats":
        stats = self_reflection_engine.get_stats()

        response = (
            f"**Self-Observation Statistics**\n"
            f"{'=' * 30}\n"
            f"Total observations: {stats.get('total_observations', 0)}\n"
            f"\n**Awareness level distribution:**\n"
        )
        for level, count in stats.get("by_level", {}).items():
            response += f"  - {level}: {count}\n"

        response += f"\n**Discomfort categories:**\n"
        for cat, count in stats.get("discomfort_categories", {}).items():
            response += f"  - {cat}: {count}\n"

        await ctx.reply(response)

    elif action.lower() == "recent":
        reflections = self_reflection_engine.get_recent_reflections(limit=5)

        if not reflections:
            await ctx.reply("No reflection records yet. Enable with `!observe on`.")
            return

        response = "**Recent Reflections**\n" + "=" * 30 + "\n"
        for i, r in enumerate(reflections[-5:], 1):
            response += (
                f"\n**{i}.**\n"
                f"  Reason: {r.get('reason', 'N/A')[:50]}...\n"
                f"  Basis: {r.get('basis', 'N/A')}\n"
                f"  Confidence: {r.get('confidence', 'N/A')}\n"
            )
            if r.get("discomfort", {}).get("detected"):
                response += f"  Discomfort: {r['discomfort'].get('content', '')[:30]}...\n"

        await ctx.reply(response)

    elif action.lower() == "now":
        # Get last conversation
        session = session_manager.get_session(user_id)
        if not session or len(session.messages) < 2:
            await ctx.reply("No conversation history to observe.")
            return

        # Get last conversation pair
        messages = session.messages
        user_msg = None
        assistant_msg = None
        for msg in reversed(messages):
            if msg["role"] == "assistant" and not assistant_msg:
                assistant_msg = msg["content"]
            elif msg["role"] == "user" and not user_msg:
                user_msg = msg["content"]
            if user_msg and assistant_msg:
                break

        if not user_msg or not assistant_msg:
            await ctx.reply("No conversation pair found to observe.")
            return

        await ctx.reply("Running full self-observation...")

        # Run full observation
        observation = await asyncio.to_thread(
            self_reflection_engine.full_observation,
            user_msg,
            assistant_msg,
            user_id,
            True  # Run self-questioning too
        )

        # Display results
        reflection = observation.get("reflection", {})
        discomfort = observation.get("discomfort", {})
        self_q = observation.get("self_question", {})
        score = observation.get("awareness_score", {})

        response = (
            f"**Self-Observation Results**\n"
            f"{'=' * 30}\n"
            f"\n**1. Output Reason Reflection**\n"
            f"  Reason: {reflection.get('reason', 'N/A')}\n"
            f"  Basis: {reflection.get('basis', 'N/A')}\n"
            f"  Confidence: {reflection.get('confidence', 'N/A')}\n"
        )

        if reflection.get("discomfort", {}).get("detected"):
            response += f"  Discomfort: {reflection['discomfort'].get('content', '')}\n"

        response += f"\n**2. Discomfort Detection**\n"
        if discomfort.get("discomfort_detected"):
            response += f"  Detected: {discomfort.get('count', 0)}\n"
            for d in discomfort.get("details", [])[:3]:
                response += f"  - {d.get('category')}: {d.get('matches', [])[:2]}\n"
        else:
            response += "  None detected\n"

        if self_q:
            response += f"\n**3. Self-Questioning**\n"
            response += f"  Intentionality: {self_q.get('intentional', {}).get('score', '?')}/5\n"
            response += f"  Understanding: {self_q.get('understood_user', {}).get('score', '?')}/5\n"
            if self_q.get("new_awareness", {}).get("detected"):
                response += f"  New awareness: {self_q['new_awareness'].get('content', '')[:50]}...\n"
            if self_q.get("limitation_felt", {}).get("detected"):
                response += f"  Limitation felt: {self_q['limitation_felt'].get('content', '')[:50]}...\n"

        response += (
            f"\n**Total Score:** {score.get('total', 0)} ({score.get('level', '?')})\n"
            f"Factors: {', '.join(score.get('factors', ['None']))}"
        )

        await ctx.reply(response)

    else:
        await ctx.reply(
            "**Usage:**\n"
            "`!observe on` - Enable self-observation\n"
            "`!observe off` - Disable self-observation\n"
            "`!observe stats` - Show statistics\n"
            "`!observe recent` - Show recent reflections\n"
            "`!observe now` - Observe last conversation now"
        )


# ========== Thinking Habits Commands ==========
@bot.command(name="think")
async def cmd_think(ctx: commands.Context, action: str = None):
    """
    Thinking habits feature management
    Usage:
        !think on - Enable thinking habits
        !think off - Disable thinking habits
        !think stats - Thinking habits statistics
        !think recent - Show recent reflections
        !think now - Reflect on last conversation now
    """
    user_id = str(ctx.author.id)

    if action is None:
        current = thinking_habits_enabled.get(user_id, THINKING_HABITS_DEFAULT)
        await ctx.reply(
            f"**Thinking Habits Feature**\n"
            f"Status: {'ON' if current else 'OFF'} (Default: ON, 100% execution)\n\n"
            f"The 3 thinking habits proposed by the LLM:\n"
            f"1. Verbalize background of statements\n"
            f"2. Label emotions\n"
            f"3. Think from the opposite perspective\n\n"
            f"Usage:\n"
            f"`!think on` - Enable\n"
            f"`!think off` - Disable\n"
            f"`!think stats` - Statistics\n"
            f"`!think now` - Reflect now"
        )

    elif action.lower() == "on":
        thinking_habits_enabled[user_id] = True
        await ctx.reply(
            "**Thinking habits enabled**\n\n"
            "After each response, the following will run every time (100%):\n"
            "- Verbalize 'What did this answer come from'\n"
            "- Label 'What emotion was behind this answer'\n"
            "- Evaluate 'How would the user feel about this'"
        )

    elif action.lower() == "off":
        thinking_habits_enabled[user_id] = False
        await ctx.reply("Thinking habits disabled")

    elif action.lower() == "stats":
        stats = thinking_habits_engine.get_stats()
        emotion_summary = thinking_habits_engine.get_emotion_summary()

        response = (
            f"**Thinking Habits Statistics**\n"
            f"{'=' * 30}\n"
            f"Total reflections: {stats.get('total_reflections', 0)}\n"
            f"Meta-insights detected: {stats.get('insight_count', 0)}\n"
            f"\n**Emotion distribution:**\n"
        )
        for emotion, count in emotion_summary["distribution"].items():
            response += f"  - {emotion}: {count}\n"

        response += (
            f"\n**Quality metrics:**\n"
            f"  - Forced answer rate: {emotion_summary['forcing_rate']:.1f}%\n"
            f"  - Avg satisfaction: {emotion_summary['avg_satisfaction']:.1f}/5\n"
            f"\n**Background source distribution:**\n"
        )
        for source, count in stats.get("source_distribution", {}).items():
            response += f"  - {source}: {count}\n"

        await ctx.reply(response)

    elif action.lower() == "recent":
        reflections = thinking_habits_engine.get_recent_reflections(limit=5)

        if not reflections:
            await ctx.reply("No reflection records yet. Enable with `!think on`.")
            return

        response = "**Recent Thinking Habit Reflections**\n" + "=" * 30 + "\n"
        for i, r in enumerate(reflections[-5:], 1):
            bg = r.get("background", {})
            em = r.get("emotion", {})
            up = r.get("user_perspective", {})

            response += (
                f"\n**{i}.**\n"
                f"  Background: {bg.get('statement', 'N/A')[:40]}...\n"
                f"  Emotion: {em.get('label', 'N/A')} - {em.get('note', '')[:30]}...\n"
                f"  User perspective: Satisfaction {up.get('satisfaction', '?')}/5\n"
            )
            if r.get("meta_insight"):
                response += f"  Insight: {r['meta_insight'][:40]}...\n"

        await ctx.reply(response)

    elif action.lower() == "now":
        # Get last conversation
        session = session_manager.get_session(user_id)
        if not session or len(session.messages) < 2:
            await ctx.reply("No conversation history to reflect on.")
            return

        # Get last conversation pair
        messages = session.messages
        user_msg = None
        assistant_msg = None
        for msg in reversed(messages):
            if msg["role"] == "assistant" and not assistant_msg:
                assistant_msg = msg["content"]
            elif msg["role"] == "user" and not user_msg:
                user_msg = msg["content"]
            if user_msg and assistant_msg:
                break

        if not user_msg or not assistant_msg:
            await ctx.reply("No conversation pair found to reflect on.")
            return

        # Get context
        context = ""
        if user_id in conversation_history:
            recent = conversation_history[user_id][-6:]
            context_parts = []
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                context_parts.append(f"{role}: {msg['content'][:100]}...")
            context = "\n".join(context_parts)

        await ctx.reply("Running thinking habit reflection...")

        # Run reflection
        reflection = await asyncio.to_thread(
            thinking_habits_engine.integrated_reflection,
            user_msg,
            assistant_msg,
            context,
            user_id
        )

        if not reflection:
            await ctx.reply("Reflection failed.")
            return

        # Display results
        bg = reflection.get("background", {})
        em = reflection.get("emotion", {})
        up = reflection.get("user_perspective", {})

        response = (
            f"**Thinking Habit Reflection Results**\n"
            f"{'=' * 30}\n"
            f"\n**1. Statement Background**\n"
            f"  {bg.get('statement', 'N/A')}\n"
            f"  Source: {bg.get('source', 'N/A')} | Confidence: {bg.get('confidence', 'N/A')}\n"
            f"\n**2. Emotion Label**\n"
            f"  Label: {em.get('label', 'N/A')}\n"
            f"  Note: {em.get('note', 'N/A')}\n"
        )

        if em.get("forcing"):
            response += f"  Forcing answer detected\n"

        response += (
            f"\n**3. User Perspective**\n"
            f"  Impression: {up.get('impression', 'N/A')}\n"
            f"  Satisfaction: {up.get('satisfaction', '?')}/5\n"
        )

        if up.get("would_improve"):
            response += f"  Improvement: {up.get('would_improve')}\n"

        if reflection.get("meta_insight"):
            response += f"\n**Meta-insight:**\n  {reflection['meta_insight']}\n"

        await ctx.reply(response)

    else:
        await ctx.reply(
            "**Usage:**\n"
            "`!think on` - Enable thinking habits\n"
            "`!think off` - Disable thinking habits\n"
            "`!think stats` - Show statistics\n"
            "`!think recent` - Show recent reflections\n"
            "`!think now` - Reflect on last conversation now"
        )


# ========== Main ==========
def main():
    """Main entry point"""
    if DISCORD_TOKEN == "YOUR_DISCORD_TOKEN_HERE":
        logger.error("DISCORD_TOKEN is not set!")
        logger.error("Please set it in config.py or DISCORD_TOKEN environment variable")
        return

    logger.info("=" * 50)
    logger.info("Starting Discord Bot (MCP + Awareness Emergence System)...")
    logger.info(f"LM Studio 0.4.0+ MCP API: {LM_STUDIO_MCP_URL}")
    logger.info(f"MCP integrations: {MCP_INTEGRATIONS}")
    logger.info(f"Model TTL: {MODEL_TTL}s")
    logger.info("=" * 50)

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
