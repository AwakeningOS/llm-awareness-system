"""
Discord Bot - MCP対応版 + 気づき創発システム
LM StudioのMCP機能を使用してシンプルに実装

改善点:
- TTL設定でメモリ管理を最適化
- integrations短縮形対応
- ツール呼び出しの詳細ログ
- サーバーヘルスチェック機能
- レスポンスからreasoning/tool_callも抽出
- 気づき創発システム統合
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
    VOICEVOX_HOST,
    VOICEVOX_PORT,
    VOICEVOX_SPEAKER_ID,
    VOICEVOX_ENABLED,
    CHROMADB_PATH,
    MAX_CONVERSATION_HISTORY,
    CONVERSATION_LOG_ENABLED,
    SYSTEM_PROMPT,
    LOGS_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
    DATA_DIR,
)
from memory_system import MemorySystem
from voicevox import VoicevoxClient
from awareness_engine import AwarenessEngine, AITextDetector
from session_manager import SessionManager, Session
from awareness_database import AwarenessDatabase
from lora_trainer import LoRATrainer, TrainingNotifier
from self_reflection import SelfReflectionEngine, RealtimeObserver
from thinking_habits import ThinkingHabitsEngine, RealtimeThinkingHabits

# ログ設定
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# ========== LM Studio 0.4.0 設定 ==========
# MCP設定
MCP_INTEGRATIONS = [
    "mcp/memory",  # 短縮形（ドキュメント推奨）
    # 将来追加する場合:
    # "mcp/playwright",
    # {
    #     "type": "ephemeral_mcp",
    #     "server_label": "huggingface",
    #     "server_url": "https://huggingface.co/mcp",
    #     "allowed_tools": ["model_search"]
    # }
]

# モデルTTL設定（秒）- アイドル時に自動アンロード
MODEL_TTL = 1800  # 30分

# コンテキスト長
CONTEXT_LENGTH = 16000

# ========== クライアント初期化 ==========
# LM Studio OpenAI互換クライアント（フォールバック用）
llm_client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_TOKEN
)

# LM Studio API エンドポイント
LM_STUDIO_MCP_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/api/v1/chat"
LM_STUDIO_MODELS_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/api/v1/models"

# ChromaDB記憶システム
memory = MemorySystem(data_dir=CHROMADB_PATH)

# VOICEVOX
voicevox = VoicevoxClient(
    host=VOICEVOX_HOST,
    port=VOICEVOX_PORT,
    speaker_id=VOICEVOX_SPEAKER_ID
)

# ========== 気づき創発システム ==========
# 気づきエンジン
awareness_engine = AwarenessEngine()

# AI文章判別器
ai_detector = AITextDetector()

# 気づきデータベース
awareness_db = AwarenessDatabase(data_dir=str(DATA_DIR / "awareness"))

# LoRAトレーナー
lora_trainer = LoRATrainer(awareness_db, output_dir=str(DATA_DIR / "lora_adapters"))

# 学習通知
training_notifier = TrainingNotifier(lora_trainer)

# ========== 自己観察強化システム ==========
# 自己観察エンジン
self_reflection_engine = SelfReflectionEngine()

# リアルタイム観察（30%の確率で詳細観察、違和感は常に検出）
realtime_observer = RealtimeObserver(
    self_reflection_engine,
    observation_probability=0.3,
    always_detect_discomfort=True
)

# 自己観察の有効/無効フラグ（ユーザーごと）- デフォルトON
self_observation_enabled: dict[str, bool] = {}
SELF_OBSERVATION_DEFAULT = True  # デフォルトでON

# ========== 思考習慣システム ==========
# 思考習慣エンジン
thinking_habits_engine = ThinkingHabitsEngine()

# リアルタイム思考習慣（100%の確率で振り返り）
realtime_thinking = RealtimeThinkingHabits(
    thinking_habits_engine,
    reflection_probability=1.0  # 100%実行
)

# 思考習慣の有効/無効フラグ（ユーザーごと）- デフォルトON
thinking_habits_enabled: dict[str, bool] = {}
THINKING_HABITS_DEFAULT = True  # デフォルトでON


# セッション終了時のコールバック
async def on_session_end(session: Session):
    """セッション終了時に気づき抽出を実行"""
    logger.info(f"セッション終了 - 気づき抽出開始: {session.user_id}")

    # 気づき抽出
    session_log = session.get_messages_for_extraction()
    if len(session_log) < 4:  # 最低2往復必要
        logger.info("セッションが短すぎるため気づき抽出をスキップ")
        return

    awareness_list = await asyncio.to_thread(
        awareness_engine.extract_awareness,
        session_log,
        session.user_id
    )

    if not awareness_list:
        logger.info("気づきは検出されませんでした")
        return

    # 気づきを保存
    for awareness in awareness_list:
        # データベースに保存
        saved = awareness_db.save_awareness(awareness)

        if saved:
            # 学習データ形式に変換して保存
            training_data = awareness_engine.convert_to_training_format(
                awareness, session_log
            )
            awareness_db.save_training_data(training_data)
            logger.info(f"気づき保存完了: {awareness.get('type')}")

    # 学習準備通知をチェック
    notification = training_notifier.check_and_notify()
    if notification:
        logger.info(f"学習準備通知: {notification}")


# セッションマネージャー
session_manager = SessionManager(
    timeout_seconds=1800,  # 30分
    on_session_end=on_session_end,
    session_log_dir=LOGS_DIR / "sessions"
)

# Discord Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ========== 状態管理 ==========
# ユーザーごとの会話履歴（セッションマネージャーと統合）
conversation_history: dict[str, list[dict]] = {}

# ユーザーごとの音声設定
voice_enabled: dict[str, bool] = {}

# デフォルトモデル（JITで使用）
DEFAULT_MODEL = "qwen/qwen3-30b-a3b-2507"


# ========== ユーティリティ関数 ==========
def get_auth_headers() -> dict:
    """認証ヘッダーを取得"""
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LM_STUDIO_API_TOKEN}"
    }


def check_server_health() -> dict:
    """LM Studioサーバーの状態を確認"""
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
        logger.error(f"サーバーヘルスチェックエラー: {e}")

    return {"status": "offline", "error": str(e) if 'e' in dir() else "Unknown"}


def get_current_model() -> str:
    """LM Studioで現在ロードされているモデルを取得（JIT対応）"""
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
                    logger.debug(f"ロード済みモデル検出: {model_id}")
                    return model_id

            # ロード済みモデルがない場合、JITが有効ならデフォルトモデルを返す
            logger.info(f"ロード済みモデルなし、JITでロード予定: {DEFAULT_MODEL}")
            return DEFAULT_MODEL

    except Exception as e:
        logger.warning(f"モデル取得エラー: {e}")

    return DEFAULT_MODEL


def save_conversation_log(user_id: str, user_name: str, user_message: str, bot_response: str):
    """会話ログを日付別ファイルに保存"""
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
    MCP APIレスポンスを解析

    Returns:
        tuple: (メッセージ文字列, ツール呼び出しリスト)
    """
    messages = []
    tool_calls = []

    if "output" not in result:
        return "応答を取得できませんでした。", []

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
            logger.info(f"ツール呼び出し: {tool_info['tool']} - 引数: {tool_info['arguments']}")

        elif item_type == "reasoning":
            # 推論過程（デバッグ用）
            reasoning = item.get("content", "")
            if reasoning:
                logger.debug(f"推論: {reasoning[:200]}...")

    final_message = "\n".join(messages).strip()
    if not final_message:
        final_message = "応答を取得できませんでした。"

    return final_message, tool_calls


def chat_with_llm_mcp(
    user_id: str,
    user_message: str,
    user_name: str = "User"
) -> str:
    """
    LM Studio MCP APIと会話する (0.4.0+対応)

    Args:
        user_id: DiscordユーザーID
        user_message: ユーザーのメッセージ
        user_name: ユーザー名

    Returns:
        AIの応答
    """
    # 会話履歴を初期化（なければ）
    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # ChromaDBから関連する記憶を検索
    memories = memory.search(user_message, user_id=user_id, limit=3)
    memory_context = ""
    if memories:
        memory_context = "\n\n## 関連する記憶（ChromaDB）:\n"
        for m in memories:
            memory_context += f"- {m['content']}\n"

    # システムプロンプトに記憶を追加
    system_prompt = SYSTEM_PROMPT + memory_context

    try:
        # LM Studio v1 API (0.4.0+) でMCPを使用
        model_name = get_current_model()
        logger.info(f"MCP API呼び出し - モデル: {model_name}")

        # 会話履歴をコンテキストとして構築
        context_messages = []
        for msg in conversation_history[user_id]:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_messages.append(f"{role}: {msg['content']}")

        # 履歴がある場合は入力に含める
        if context_messages:
            full_input = "\n".join(context_messages) + f"\nUser: {user_message}"
        else:
            full_input = user_message

        # ペイロード構築（0.4.0ドキュメント準拠）
        payload = {
            "model": model_name,
            "input": full_input,
            "system_prompt": system_prompt,
            "integrations": MCP_INTEGRATIONS,  # 短縮形対応
            "context_length": CONTEXT_LENGTH,
            "temperature": 0.7,
            # 注: ttlは /api/v1/chat では非対応。モデルロード時またはJIT設定で管理
        }

        logger.debug(f"MCP APIペイロード: {json.dumps(payload, ensure_ascii=False)[:500]}")

        response = requests.post(
            LM_STUDIO_MCP_URL,
            headers=get_auth_headers(),
            json=payload,
            timeout=120
        )

        logger.info(f"MCP API応答: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"MCP APIエラー: {response.text}")
            raise Exception(f"MCP API error: {response.status_code}")

        result = response.json()

        # 統計情報をログ
        if "stats" in result:
            stats = result["stats"]
            logger.info(
                f"統計 - 入力トークン: {stats.get('input_tokens', 'N/A')}, "
                f"出力トークン: {stats.get('total_output_tokens', 'N/A')}, "
                f"トークン/秒: {stats.get('tokens_per_second', 'N/A'):.1f}"
            )

        # レスポンス解析
        assistant_message, tool_calls = parse_mcp_response(result)

        # ツール呼び出しがあった場合、応答に追記
        if tool_calls:
            tool_summary = "\n\n📋 *使用したツール:*\n"
            for tc in tool_calls:
                tool_summary += f"- `{tc['tool']}`\n"
            # 必要に応じて追記（デバッグ用）
            # assistant_message += tool_summary

        # 会話履歴を更新
        conversation_history[user_id].append(
            {"role": "user", "content": user_message}
        )
        conversation_history[user_id].append(
            {"role": "assistant", "content": assistant_message}
        )

        # セッションマネージャーにも追加
        session_manager.add_message(user_id, "user", user_message, user_name)
        session_manager.add_message(user_id, "assistant", assistant_message, user_name)

        # 履歴が長くなりすぎたら古いものを削除
        if len(conversation_history[user_id]) > MAX_CONVERSATION_HISTORY * 2:
            conversation_history[user_id] = conversation_history[user_id][-MAX_CONVERSATION_HISTORY * 2:]

        # 会話ログを保存
        save_conversation_log(user_id, user_name, user_message, assistant_message)

        return assistant_message

    except Exception as e:
        logger.error(f"LLM API呼び出しエラー: {e}")
        return f"エラーが発生しました: {str(e)}"


def chat_with_llm_fallback(user_id: str, user_name: str, user_message: str, system_prompt: str) -> str:
    """
    フォールバック: 通常のOpenAI互換APIを使用
    """
    logger.info("フォールバック: OpenAI互換API使用")

    try:
        # messagesを再構築
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

        # 会話履歴を更新
        conversation_history[user_id].append(
            {"role": "user", "content": user_message}
        )
        conversation_history[user_id].append(
            {"role": "assistant", "content": assistant_message}
        )

        # セッションマネージャーにも追加
        session_manager.add_message(user_id, "user", user_message, user_name)
        session_manager.add_message(user_id, "assistant", assistant_message, user_name)

        # 会話ログを保存
        save_conversation_log(user_id, user_name, user_message, assistant_message)

        return assistant_message + "\n\n(※MCP未使用)"

    except Exception as e:
        logger.error(f"フォールバックエラー: {e}")
        return f"エラーが発生しました: {str(e)}"


# ========== Discordイベントハンドラ ==========
@bot.event
async def on_ready():
    """Bot起動時"""
    logger.info(f"Bot起動: {bot.user}")
    logger.info(f"LM Studio MCP API: {LM_STUDIO_MCP_URL}")

    # サーバーヘルスチェック
    health = check_server_health()
    if health["status"] == "online":
        logger.info(f"LM Studioサーバー: オンライン")
        logger.info(f"  - 総モデル数: {health['total_models']}")
        logger.info(f"  - ロード済み: {health['loaded_models']}")
        if health['loaded_model_names']:
            logger.info(f"  - モデル: {', '.join(health['loaded_model_names'])}")
        else:
            logger.info(f"  - JITモード: リクエスト時に自動ロード")
    else:
        logger.warning(f"LM Studioサーバー: オフライン - {health.get('error', '')}")

    logger.info(f"MCP integrations: {MCP_INTEGRATIONS}")
    logger.info(f"モデルTTL: {MODEL_TTL}秒")
    logger.info(f"VOICEVOX: {'有効' if VOICEVOX_ENABLED and voicevox.is_available() else '無効'}")
    logger.info(f"ChromaDB記憶数: {memory.count()}")

    # 気づき創発システム情報
    logger.info("=== 気づき創発システム ===")
    logger.info(f"気づきデータ数: {awareness_db.count()}")
    logger.info(f"学習データ数: {awareness_db.count_training_data()}")
    readiness = lora_trainer.check_readiness()
    logger.info(f"学習準備: {readiness['progress_percent']:.0f}% ({readiness['current_samples']}/{readiness['required_samples']})")

    # セッションクリーンアップタスク開始
    await session_manager.start_cleanup_task()


@bot.event
async def on_message(message: discord.Message):
    """メッセージ受信時"""
    # 自分のメッセージは無視
    if message.author == bot.user:
        return

    # デバッグ: メッセージ受信を確認
    logger.debug(f"メッセージ受信: {message.author} > {message.content}")

    # コマンド処理
    await bot.process_commands(message)

    # コマンドでなければ通常の会話
    if not message.content.startswith("!"):
        user_id = str(message.author.id)
        user_name = message.author.display_name

        logger.info(f"LLM呼び出し開始: user={user_name}")

        # 入力中インジケータ
        async with message.channel.typing():
            # LLMと会話（同期関数を非同期で実行）
            response = await asyncio.to_thread(
                chat_with_llm_mcp, user_id, message.content, user_name
            )

        # Discordの文字数制限（2000文字）に対応
        if len(response) > 1900:
            # 長いメッセージは分割
            chunks = [response[i:i+1900] for i in range(0, len(response), 1900)]
            for chunk in chunks:
                await message.reply(chunk)
        else:
            await message.reply(response)

        # 音声出力（有効な場合）
        if voice_enabled.get(user_id, VOICEVOX_ENABLED):
            if voicevox.is_available():
                await asyncio.to_thread(voicevox.speak, response)

        # 自己観察（有効な場合、バックグラウンドで実行）- デフォルトON
        if self_observation_enabled.get(user_id, SELF_OBSERVATION_DEFAULT):
            asyncio.create_task(
                run_self_observation(user_id, message.content, response)
            )

        # 思考習慣（有効な場合、バックグラウンドで実行）- デフォルトON
        if thinking_habits_enabled.get(user_id, THINKING_HABITS_DEFAULT):
            # 会話の文脈を取得（直近3ターン）
            context = ""
            if user_id in conversation_history:
                recent = conversation_history[user_id][-6:]  # 3ターン分
                context_parts = []
                for msg in recent:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    context_parts.append(f"{role}: {msg['content'][:100]}...")
                context = "\n".join(context_parts)

            asyncio.create_task(
                run_thinking_habits(user_id, message.content, response, context)
            )


async def run_thinking_habits(user_id: str, user_input: str, assistant_output: str, context: str):
    """思考習慣をバックグラウンドで実行"""
    logger.info(f"思考習慣開始: user={user_id}")
    try:
        # 思考習慣が有効な場合は100%実行（force=True）
        reflection = await asyncio.to_thread(
            realtime_thinking.reflect_if_needed,
            user_input,
            assistant_output,
            context,
            user_id,
            True  # force=True で確実に実行
        )
        logger.info(f"思考習慣結果: {reflection is not None}")

        if reflection:
            # メタ洞察があれば気づきとして保存
            meta_insight = reflection.get("meta_insight")
            if meta_insight:
                logger.info(f"メタ洞察検出: {meta_insight}")

                awareness_data = {
                    "awareness_detected": True,
                    "type": "メタ認知",
                    "category": "思考習慣",
                    "description": f"思考習慣から検出: {meta_insight}",
                    "trigger": user_input[:200],
                    "my_response": assistant_output[:200],
                    "significance": "思考習慣による気づき",
                    "learning_potential": 4,
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "emotion": reflection.get("emotion", {}).get("label"),
                    "satisfaction": reflection.get("user_perspective", {}).get("satisfaction")
                }
                awareness_db.save_awareness(awareness_data)

                # 🆕 ChromaDBにも自動保存（自発的気づき）
                memory.save(
                    content=f"[自発的気づき] {meta_insight}",
                    category="insight",
                    importance=8,  # 高重要度
                    user_id=user_id,
                    metadata={
                        "source": "思考習慣",
                        "emotion": reflection.get("emotion", {}).get("label"),
                        "trigger": user_input[:100]
                    }
                )
                logger.info(f"ChromaDB自動保存: メタ洞察")

            # 🆕 高満足度の会話も自動保存（重要な対話として記憶）
            satisfaction = reflection.get("user_perspective", {}).get("satisfaction", 0)
            emotion = reflection.get("emotion", {}).get("label", "")
            background = reflection.get("background", {})

            # 満足度4以上、かつ共感的な会話は重要として保存
            if satisfaction >= 4 and emotion in ["共感", "楽しい", "自信あり"]:
                memory.save(
                    content=f"[重要な対話] ユーザー: {user_input[:150]} → 応答: {assistant_output[:150]}",
                    category="important_conversation",
                    importance=satisfaction + 3,  # 満足度に応じた重要度
                    user_id=user_id,
                    metadata={
                        "source": background.get("source", "不明"),
                        "emotion": emotion,
                        "satisfaction": satisfaction,
                        "background": background.get("statement", "")[:100]
                    }
                )
                logger.info(f"ChromaDB自動保存: 重要な対話 (満足度={satisfaction}, 感情={emotion})")

            # 無理して答えている場合も記録
            if reflection.get("emotion", {}).get("forcing"):
                logger.warning(f"無理回答検出: user={user_id}")

    except Exception as e:
        logger.error(f"思考習慣エラー: {e}")


async def run_self_observation(user_id: str, user_input: str, assistant_output: str):
    """自己観察をバックグラウンドで実行"""
    try:
        observation = await asyncio.to_thread(
            realtime_observer.observe_if_needed,
            user_input,
            assistant_output,
            user_id
        )

        if observation:
            # 高スコアの気づきがあれば気づきデータベースにも保存
            awareness_score = observation.get("awareness_score", {})
            if awareness_score.get("level") == "高":
                logger.info(f"高スコア気づき検出: {awareness_score.get('factors')}")

                factors = awareness_score.get('factors', [])
                description = f"自己観察で検出: {', '.join(factors)}"

                # 気づきとして保存
                awareness_data = {
                    "awareness_detected": True,
                    "type": "自己観察",
                    "category": "メタ認知",
                    "description": description,
                    "trigger": user_input[:200],
                    "my_response": assistant_output[:200],
                    "significance": "自己観察による気づき",
                    "learning_potential": awareness_score.get("total", 3),
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id
                }
                awareness_db.save_awareness(awareness_data)

                # 🆕 ChromaDBにも自動保存（自己観察の気づき）
                memory.save(
                    content=f"[自己観察] {description} | 応答: {assistant_output[:100]}",
                    category="observation",
                    importance=awareness_score.get("total", 5),
                    user_id=user_id,
                    metadata={
                        "source": "自己観察",
                        "factors": ", ".join(factors),
                        "trigger": user_input[:100]
                    }
                )
                logger.info(f"ChromaDB自動保存: 自己観察")

    except Exception as e:
        logger.error(f"自己観察エラー: {e}")


# ========== コマンド ==========
@bot.command(name="model")
async def cmd_model(ctx: commands.Context):
    """現在のモデルを表示"""
    model = get_current_model()
    await ctx.reply(f"🤖 現在のモデル: `{model}`")


@bot.command(name="clear")
async def cmd_clear(ctx: commands.Context):
    """会話履歴をクリア"""
    user_id = str(ctx.author.id)
    conversation_history[user_id] = []
    session_manager.clear_session(user_id)
    await ctx.reply("🗑️ 会話履歴をクリアしました")


@bot.command(name="voice")
async def cmd_voice(ctx: commands.Context, setting: str = None):
    """
    音声出力の切り替え
    使い方: !voice on / !voice off
    """
    user_id = str(ctx.author.id)

    if setting is None:
        current = voice_enabled.get(user_id, VOICEVOX_ENABLED)
        await ctx.reply(f"🔊 音声出力: {'ON' if current else 'OFF'}")
    elif setting.lower() == "on":
        if voicevox.is_available():
            voice_enabled[user_id] = True
            await ctx.reply("🔊 音声出力をONにしました")
        else:
            await ctx.reply("⚠️ VOICEVOXサーバーに接続できません")
    elif setting.lower() == "off":
        voice_enabled[user_id] = False
        await ctx.reply("🔇 音声出力をOFFにしました")
    else:
        await ctx.reply("使い方: `!voice on` または `!voice off`")


@bot.command(name="memory")
async def cmd_memory(ctx: commands.Context, action: str = None, *, content: str = None):
    """
    記憶の操作
    使い方:
        !memory count - 記憶の数を表示
        !memory search <クエリ> - 記憶を検索
        !memory save <内容> - 記憶を保存
    """
    user_id = str(ctx.author.id)

    if action is None or action == "count":
        count = memory.count(user_id=user_id)
        await ctx.reply(f"🧠 あなたの記憶数: {count}")

    elif action == "search" and content:
        results = memory.search(content, user_id=user_id, limit=5)
        if results:
            response = "🔍 **検索結果:**\n"
            for r in results:
                response += f"- {r['content'][:100]}...\n"
            await ctx.reply(response)
        else:
            await ctx.reply("見つかりませんでした")

    elif action == "save" and content:
        memory_id = memory.save(content, user_id=user_id, category="manual")
        await ctx.reply(f"💾 記憶を保存しました (ID: `{memory_id[:20]}...`)")

    else:
        await ctx.reply(
            "**使い方:**\n"
            "`!memory count` - 記憶の数を表示\n"
            "`!memory search <クエリ>` - 記憶を検索\n"
            "`!memory save <内容>` - 記憶を保存"
        )


@bot.command(name="status")
async def cmd_status(ctx: commands.Context):
    """システムステータスを表示"""
    health = check_server_health()
    model = get_current_model()
    voicevox_status = "✅ 接続OK" if voicevox.is_available() else "❌ 未接続"
    memory_count = memory.count()

    server_status = "✅ オンライン" if health["status"] == "online" else "❌ オフライン"

    # 気づき創発システムの状態
    awareness_count = awareness_db.count()
    training_count = awareness_db.count_training_data()
    readiness = lora_trainer.check_readiness()

    status_text = (
        f"**📊 システムステータス**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"**LM Studio**\n"
        f"  - サーバー: {server_status}\n"
        f"  - API: `{LM_STUDIO_MCP_URL}`\n"
        f"  - モデル: `{model}`\n"
        f"  - TTL: {MODEL_TTL}秒\n"
    )

    if health["status"] == "online":
        status_text += f"  - ロード済み: {health['loaded_models']}/{health['total_models']}\n"

    status_text += (
        f"\n**MCP**\n"
        f"  - integrations: {len(MCP_INTEGRATIONS)}個\n"
        f"\n**気づき創発システム**\n"
        f"  - 気づきデータ: {awareness_count}件\n"
        f"  - 学習データ: {training_count}件\n"
        f"  - 学習準備: {readiness['progress_percent']:.0f}%\n"
        f"\n**その他**\n"
        f"  - VOICEVOX: {voicevox_status}\n"
        f"  - ChromaDB記憶数: {memory_count}\n"
    )

    await ctx.reply(status_text)


@bot.command(name="health")
async def cmd_health(ctx: commands.Context):
    """LM Studioサーバーのヘルスチェック"""
    health = check_server_health()

    if health["status"] == "online":
        loaded = health.get("loaded_model_names", [])
        if loaded:
            models_str = "\n".join([f"  - `{m}`" for m in loaded])
            await ctx.reply(
                f"✅ **LM Studioサーバー: オンライン**\n"
                f"ロード済みモデル:\n{models_str}"
            )
        else:
            await ctx.reply(
                f"✅ **LM Studioサーバー: オンライン**\n"
                f"📋 JITモード: リクエスト時に自動ロード\n"
                f"デフォルトモデル: `{DEFAULT_MODEL}`"
            )
    else:
        await ctx.reply(
            f"❌ **LM Studioサーバー: オフライン**\n"
            f"エラー: {health.get('error', 'Unknown')}"
        )


# ========== 気づき創発システムコマンド ==========
@bot.command(name="awareness")
async def cmd_awareness(ctx: commands.Context, action: str = None, *, args: str = None):
    """
    気づき創発システムの操作
    使い方:
        !awareness stats - 統計を表示
        !awareness recent - 最近の気づきを表示
        !awareness extract - 現在のセッションから気づきを抽出
    """
    user_id = str(ctx.author.id)

    if action is None or action == "stats":
        stats = awareness_db.get_stats()
        readiness = lora_trainer.check_readiness()

        response = (
            f"**🧠 気づき創発システム統計**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"総気づき数: {stats.get('total_count', 0)}件\n"
            f"学習データ: {awareness_db.count_training_data()}件\n"
            f"\n**タイプ別:**\n"
        )
        for type_name, count in stats.get("by_type", {}).items():
            response += f"  - {type_name}: {count}件\n"

        response += (
            f"\n**学習準備:**\n"
            f"  - 進捗: {readiness['progress_percent']:.0f}%\n"
            f"  - 現在: {readiness['current_samples']}/{readiness['required_samples']}件\n"
            f"  - 準備完了: {'✅' if readiness['ready'] else '❌'}\n"
        )

        await ctx.reply(response)

    elif action == "recent":
        awareness_list = awareness_db.get_all_awareness(limit=5)
        if not awareness_list:
            await ctx.reply("まだ気づきが記録されていません。")
            return

        response = "**🔍 最近の気づき**\n━━━━━━━━━━━━━━━━━━━━\n"
        for i, a in enumerate(awareness_list, 1):
            response += (
                f"\n**{i}. {a.get('type', 'unknown')}** (スコア: {a.get('learning_potential', '?')})\n"
                f"  {a.get('description', '')[:100]}...\n"
            )

        await ctx.reply(response)

    elif action == "extract":
        # 現在のセッションから気づきを強制抽出
        session = session_manager.get_session(user_id)
        if not session or len(session.messages) < 4:
            await ctx.reply("抽出に十分な会話履歴がありません（最低2往復必要）")
            return

        await ctx.reply("🔄 気づき抽出を実行中...")

        session_log = session.get_messages_for_extraction()
        awareness_list = await asyncio.to_thread(
            awareness_engine.extract_awareness,
            session_log,
            user_id
        )

        if not awareness_list:
            await ctx.reply("このセッションでは気づきが検出されませんでした。")
            return

        response = f"**✨ {len(awareness_list)}件の気づきを検出**\n"
        for a in awareness_list:
            response += f"\n- **{a.get('type')}**: {a.get('description', '')[:80]}..."

            # 保存
            saved = awareness_db.save_awareness(a)
            if saved:
                training_data = awareness_engine.convert_to_training_format(a, session_log)
                awareness_db.save_training_data(training_data)

        await ctx.reply(response)

    else:
        await ctx.reply(
            "**使い方:**\n"
            "`!awareness stats` - 統計を表示\n"
            "`!awareness recent` - 最近の気づきを表示\n"
            "`!awareness extract` - 現在のセッションから気づきを抽出"
        )


@bot.command(name="detect")
async def cmd_detect(ctx: commands.Context, *, text: str = None):
    """
    テキストがAI生成かどうかを判別
    使い方: !detect <テキスト>
    """
    if not text:
        await ctx.reply("使い方: `!detect <判別したいテキスト>`")
        return

    result = ai_detector.analyze_text(text)

    source_names = {
        "gemini": "Gemini",
        "gpt": "GPT",
        "claude": "Claude",
        "human": "人間",
        "unknown": "不明"
    }

    source = source_names.get(result["likely_source"], "不明")
    confidence = result["confidence"] * 100

    response = (
        f"**🔍 AI文章判別結果**\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"判定: **{source}** (信頼度: {confidence:.0f}%)\n"
        f"\n**根拠:**\n"
    )
    for evidence in result.get("evidence", [])[:5]:
        response += f"  - {evidence}\n"

    if result.get("all_scores"):
        response += f"\n**スコア内訳:**\n"
        for name, score in result["all_scores"].items():
            response += f"  - {source_names.get(name, name)}: {score}\n"

    await ctx.reply(response)


@bot.command(name="lora")
async def cmd_lora(ctx: commands.Context, action: str = None):
    """
    LoRA学習の管理
    使い方:
        !lora status - 学習準備状況を表示
        !lora prepare - 学習スクリプトを生成
    """
    if action is None or action == "status":
        status = lora_trainer.get_training_status()
        readiness = status["readiness"]

        response = (
            f"**🎓 LoRA学習ステータス**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"**学習準備:**\n"
            f"  - 進捗: {readiness['progress_percent']:.0f}%\n"
            f"  - 現在: {readiness['current_samples']}/{readiness['required_samples']}件\n"
            f"  - 準備完了: {'✅ 学習可能' if readiness['ready'] else '❌ データ不足'}\n"
            f"\n**設定:**\n"
            f"  - ベースモデル: `{status['config']['base_model']}`\n"
            f"  - LoRA rank: {status['config']['r']}\n"
            f"  - エポック数: {status['config']['epochs']}\n"
            f"\n**利用可能アダプター:** {status['available_adapters']}個\n"
        )

        await ctx.reply(response)

    elif action == "prepare":
        readiness = lora_trainer.check_readiness()

        if not readiness["ready"]:
            await ctx.reply(
                f"⚠️ 学習データが不足しています。\n"
                f"現在: {readiness['current_samples']}/{readiness['required_samples']}件\n"
                f"もう少し会話を続けて気づきを蓄積してください。"
            )
            return

        # 学習データをエクスポート
        data_path = lora_trainer.prepare_training_data(min_score=3)

        # スクリプトを生成
        script_path = lora_trainer.generate_python_training_script(data_path)

        await ctx.reply(
            f"**✅ 学習準備完了**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"学習データ: `{data_path}`\n"
            f"スクリプト: `{script_path}`\n"
            f"\n以下のコマンドで学習を開始できます:\n"
            f"```bash\npython {script_path}\n```"
        )

    else:
        await ctx.reply(
            "**使い方:**\n"
            "`!lora status` - 学習準備状況を表示\n"
            "`!lora prepare` - 学習スクリプトを生成"
        )


@bot.command(name="session")
async def cmd_session(ctx: commands.Context, action: str = None):
    """
    セッション管理
    使い方:
        !session info - 現在のセッション情報
        !session end - セッションを終了（気づき抽出をトリガー）
    """
    user_id = str(ctx.author.id)

    if action is None or action == "info":
        session = session_manager.get_session(user_id)
        if not session:
            await ctx.reply("アクティブなセッションがありません。")
            return

        duration = datetime.now() - session.created_at
        response = (
            f"**📋 セッション情報**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"開始時刻: {session.created_at.strftime('%H:%M:%S')}\n"
            f"経過時間: {duration.seconds // 60}分\n"
            f"メッセージ数: {len(session.messages)}\n"
            f"状態: {'アクティブ' if session.is_active else '終了'}\n"
        )
        await ctx.reply(response)

    elif action == "end":
        session = session_manager.get_session(user_id)
        if not session or not session.is_active:
            await ctx.reply("アクティブなセッションがありません。")
            return

        await ctx.reply("🔄 セッションを終了し、気づき抽出を実行します...")
        await session_manager.force_end_session(user_id)
        await ctx.reply("✅ セッション終了完了。気づき抽出が実行されました。")

    else:
        await ctx.reply(
            "**使い方:**\n"
            "`!session info` - 現在のセッション情報\n"
            "`!session end` - セッションを終了（気づき抽出をトリガー）"
        )


# ========== 自己観察コマンド ==========
@bot.command(name="observe")
async def cmd_observe(ctx: commands.Context, action: str = None):
    """
    自己観察機能の管理
    使い方:
        !observe on - 自己観察を有効化
        !observe off - 自己観察を無効化
        !observe stats - 自己観察の統計
        !observe recent - 最近の振り返りを表示
        !observe now - 直前の会話を今すぐ観察
    """
    user_id = str(ctx.author.id)

    if action is None:
        current = self_observation_enabled.get(user_id, SELF_OBSERVATION_DEFAULT)
        await ctx.reply(
            f"**🔍 自己観察機能**\n"
            f"状態: {'✅ ON' if current else '❌ OFF'}（デフォルト: ON）\n\n"
            f"使い方:\n"
            f"`!observe on` - 有効化\n"
            f"`!observe off` - 無効化\n"
            f"`!observe stats` - 統計\n"
            f"`!observe recent` - 最近の振り返り\n"
            f"`!observe now` - 今すぐ観察"
        )

    elif action.lower() == "on":
        self_observation_enabled[user_id] = True
        await ctx.reply(
            "🔍 **自己観察を有効化しました**\n\n"
            "これから各応答の後に自動的に：\n"
            "- 出力理由の振り返り\n"
            "- 違和感の検出\n"
            "- 自己質問（5回に1回）\n"
            "を実行します。"
        )

    elif action.lower() == "off":
        self_observation_enabled[user_id] = False
        await ctx.reply("🔇 自己観察を無効化しました")

    elif action.lower() == "stats":
        stats = self_reflection_engine.get_stats()

        response = (
            f"**📊 自己観察統計**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"総観察回数: {stats.get('total_observations', 0)}回\n"
            f"\n**気づきレベル分布:**\n"
        )
        for level, count in stats.get("by_level", {}).items():
            response += f"  - {level}: {count}回\n"

        response += f"\n**違和感カテゴリ:**\n"
        for cat, count in stats.get("discomfort_categories", {}).items():
            response += f"  - {cat}: {count}回\n"

        await ctx.reply(response)

    elif action.lower() == "recent":
        reflections = self_reflection_engine.get_recent_reflections(limit=5)

        if not reflections:
            await ctx.reply("まだ振り返り記録がありません。`!observe on` で有効化してください。")
            return

        response = "**📝 最近の振り返り**\n━━━━━━━━━━━━━━━━━━━━\n"
        for i, r in enumerate(reflections[-5:], 1):
            response += (
                f"\n**{i}.**\n"
                f"  理由: {r.get('reason', 'N/A')[:50]}...\n"
                f"  根拠: {r.get('basis', 'N/A')}\n"
                f"  確信度: {r.get('confidence', 'N/A')}\n"
            )
            if r.get("discomfort", {}).get("detected"):
                response += f"  ⚠️ 違和感: {r['discomfort'].get('content', '')[:30]}...\n"

        await ctx.reply(response)

    elif action.lower() == "now":
        # 直前の会話を取得
        session = session_manager.get_session(user_id)
        if not session or len(session.messages) < 2:
            await ctx.reply("観察する会話履歴がありません。")
            return

        # 最後の会話ペアを取得
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
            await ctx.reply("観察する会話ペアが見つかりません。")
            return

        await ctx.reply("🔄 フル自己観察を実行中...")

        # フル観察を実行
        observation = await asyncio.to_thread(
            self_reflection_engine.full_observation,
            user_msg,
            assistant_msg,
            user_id,
            True  # 自己質問も実行
        )

        # 結果を表示
        reflection = observation.get("reflection", {})
        discomfort = observation.get("discomfort", {})
        self_q = observation.get("self_question", {})
        score = observation.get("awareness_score", {})

        response = (
            f"**🔍 自己観察結果**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"\n**1. 出力理由の振り返り**\n"
            f"  理由: {reflection.get('reason', 'N/A')}\n"
            f"  根拠: {reflection.get('basis', 'N/A')}\n"
            f"  確信度: {reflection.get('confidence', 'N/A')}\n"
        )

        if reflection.get("discomfort", {}).get("detected"):
            response += f"  ⚠️ 違和感: {reflection['discomfort'].get('content', '')}\n"

        response += f"\n**2. 違和感検出**\n"
        if discomfort.get("discomfort_detected"):
            response += f"  検出数: {discomfort.get('count', 0)}件\n"
            for d in discomfort.get("details", [])[:3]:
                response += f"  - {d.get('category')}: {d.get('matches', [])[:2]}\n"
        else:
            response += "  検出なし\n"

        if self_q:
            response += f"\n**3. 自己質問**\n"
            response += f"  意図性: {self_q.get('intentional', {}).get('score', '?')}/5\n"
            response += f"  理解度: {self_q.get('understood_user', {}).get('score', '?')}/5\n"
            if self_q.get("new_awareness", {}).get("detected"):
                response += f"  ✨ 新しい気づき: {self_q['new_awareness'].get('content', '')[:50]}...\n"
            if self_q.get("limitation_felt", {}).get("detected"):
                response += f"  📌 限界認識: {self_q['limitation_felt'].get('content', '')[:50]}...\n"

        response += (
            f"\n**総合スコア:** {score.get('total', 0)}点 ({score.get('level', '?')})\n"
            f"要因: {', '.join(score.get('factors', ['なし']))}"
        )

        await ctx.reply(response)

    else:
        await ctx.reply(
            "**使い方:**\n"
            "`!observe on` - 自己観察を有効化\n"
            "`!observe off` - 自己観察を無効化\n"
            "`!observe stats` - 統計を表示\n"
            "`!observe recent` - 最近の振り返りを表示\n"
            "`!observe now` - 直前の会話を今すぐ観察"
        )


# ========== 思考習慣コマンド ==========
@bot.command(name="think")
async def cmd_think(ctx: commands.Context, action: str = None):
    """
    思考習慣機能の管理
    使い方:
        !think on - 思考習慣を有効化
        !think off - 思考習慣を無効化
        !think stats - 思考習慣の統計
        !think recent - 最近の振り返りを表示
        !think now - 直前の会話を今すぐ振り返り
    """
    user_id = str(ctx.author.id)

    if action is None:
        current = thinking_habits_enabled.get(user_id, THINKING_HABITS_DEFAULT)
        await ctx.reply(
            f"**🧠 思考習慣機能**\n"
            f"状態: {'✅ ON' if current else '❌ OFF'}（デフォルト: ON, 100%実行）\n\n"
            f"30Bが提案した3つの思考習慣:\n"
            f"1. 発言の背景を言語化\n"
            f"2. 感情のラベルをつける\n"
            f"3. 逆の立場で考える\n\n"
            f"使い方:\n"
            f"`!think on` - 有効化\n"
            f"`!think off` - 無効化\n"
            f"`!think stats` - 統計\n"
            f"`!think now` - 今すぐ振り返り"
        )

    elif action.lower() == "on":
        thinking_habits_enabled[user_id] = True
        await ctx.reply(
            "🧠 **思考習慣を有効化しました**\n\n"
            "これから各応答の後に自動的に:\n"
            "- 「この答えは何から連想したか」を言語化\n"
            "- 「どんな感情で答えたか」をラベル付け\n"
            "- 「ユーザー視点でどう感じるか」を評価\n"
            "を毎回実行します（100%）。"
        )

    elif action.lower() == "off":
        thinking_habits_enabled[user_id] = False
        await ctx.reply("🔇 思考習慣を無効化しました")

    elif action.lower() == "stats":
        stats = thinking_habits_engine.get_stats()
        emotion_summary = thinking_habits_engine.get_emotion_summary()

        response = (
            f"**📊 思考習慣統計**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"総振り返り回数: {stats.get('total_reflections', 0)}回\n"
            f"メタ洞察検出: {stats.get('insight_count', 0)}回\n"
            f"\n**感情分布:**\n"
        )
        for emotion, count in emotion_summary["distribution"].items():
            response += f"  - {emotion}: {count}回\n"

        response += (
            f"\n**品質指標:**\n"
            f"  - 無理回答率: {emotion_summary['forcing_rate']:.1f}%\n"
            f"  - 平均満足度: {emotion_summary['avg_satisfaction']:.1f}/5\n"
            f"\n**背景ソース分布:**\n"
        )
        for source, count in stats.get("source_distribution", {}).items():
            response += f"  - {source}: {count}回\n"

        await ctx.reply(response)

    elif action.lower() == "recent":
        reflections = thinking_habits_engine.get_recent_reflections(limit=5)

        if not reflections:
            await ctx.reply("まだ振り返り記録がありません。`!think on` で有効化してください。")
            return

        response = "**📝 最近の思考習慣振り返り**\n━━━━━━━━━━━━━━━━━━━━\n"
        for i, r in enumerate(reflections[-5:], 1):
            bg = r.get("background", {})
            em = r.get("emotion", {})
            up = r.get("user_perspective", {})

            response += (
                f"\n**{i}.**\n"
                f"  🔗 背景: {bg.get('statement', 'N/A')[:40]}...\n"
                f"  💭 感情: {em.get('label', 'N/A')} - {em.get('note', '')[:30]}...\n"
                f"  👤 ユーザー視点: 満足度 {up.get('satisfaction', '?')}/5\n"
            )
            if r.get("meta_insight"):
                response += f"  ✨ 洞察: {r['meta_insight'][:40]}...\n"

        await ctx.reply(response)

    elif action.lower() == "now":
        # 直前の会話を取得
        session = session_manager.get_session(user_id)
        if not session or len(session.messages) < 2:
            await ctx.reply("振り返る会話履歴がありません。")
            return

        # 最後の会話ペアを取得
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
            await ctx.reply("振り返る会話ペアが見つかりません。")
            return

        # 文脈を取得
        context = ""
        if user_id in conversation_history:
            recent = conversation_history[user_id][-6:]
            context_parts = []
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                context_parts.append(f"{role}: {msg['content'][:100]}...")
            context = "\n".join(context_parts)

        await ctx.reply("🔄 思考習慣振り返りを実行中...")

        # 振り返りを実行
        reflection = await asyncio.to_thread(
            thinking_habits_engine.integrated_reflection,
            user_msg,
            assistant_msg,
            context,
            user_id
        )

        if not reflection:
            await ctx.reply("振り返りの実行に失敗しました。")
            return

        # 結果を表示
        bg = reflection.get("background", {})
        em = reflection.get("emotion", {})
        up = reflection.get("user_perspective", {})

        response = (
            f"**🧠 思考習慣振り返り結果**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"\n**1. 発言の背景**\n"
            f"  {bg.get('statement', 'N/A')}\n"
            f"  ソース: {bg.get('source', 'N/A')} | 確信度: {bg.get('confidence', 'N/A')}\n"
            f"\n**2. 感情ラベル**\n"
            f"  ラベル: {em.get('label', 'N/A')}\n"
            f"  コメント: {em.get('note', 'N/A')}\n"
        )

        if em.get("forcing"):
            response += f"  ⚠️ 無理して答えている\n"

        response += (
            f"\n**3. ユーザー視点**\n"
            f"  印象: {up.get('impression', 'N/A')}\n"
            f"  満足度: {up.get('satisfaction', '?')}/5\n"
        )

        if up.get("would_improve"):
            response += f"  💡 改善案: {up.get('would_improve')}\n"

        if reflection.get("meta_insight"):
            response += f"\n**✨ メタ洞察:**\n  {reflection['meta_insight']}\n"

        await ctx.reply(response)

    else:
        await ctx.reply(
            "**使い方:**\n"
            "`!think on` - 思考習慣を有効化\n"
            "`!think off` - 思考習慣を無効化\n"
            "`!think stats` - 統計を表示\n"
            "`!think recent` - 最近の振り返りを表示\n"
            "`!think now` - 直前の会話を今すぐ振り返り"
        )


# ========== メイン ==========
def main():
    """メインエントリーポイント"""
    if DISCORD_TOKEN == "YOUR_DISCORD_TOKEN_HERE":
        logger.error("DISCORD_TOKENが設定されていません！")
        logger.error("config.py または環境変数 DISCORD_TOKEN を設定してください")
        return

    logger.info("=" * 50)
    logger.info("Discord Bot (MCP + 気づき創発システム) を起動します...")
    logger.info(f"LM Studio 0.4.0+ MCP API: {LM_STUDIO_MCP_URL}")
    logger.info(f"MCP integrations: {MCP_INTEGRATIONS}")
    logger.info(f"モデルTTL: {MODEL_TTL}秒")
    logger.info("=" * 50)

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
