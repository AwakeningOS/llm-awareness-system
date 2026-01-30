"""
設定ファイル（サンプル）
このファイルをコピーして config.py を作成し、各値を設定してください。
"""

import os
from pathlib import Path

# ========== パス設定 ==========
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
WORKSPACE_DIR = BASE_DIR / "workspace"
LOGS_DIR = BASE_DIR / "logs"

# ========== Discord設定 ==========
# Discord Developer Portal で取得したトークン
DISCORD_TOKEN = "YOUR_DISCORD_BOT_TOKEN_HERE"

# ========== LM Studio設定 ==========
LM_STUDIO_HOST = os.getenv("LM_STUDIO_HOST", "localhost")
LM_STUDIO_PORT = int(os.getenv("LM_STUDIO_PORT", "1234"))
LM_STUDIO_BASE_URL = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1"
# LM Studio の設定で API キーを有効にしている場合のみ設定
LM_STUDIO_API_TOKEN = "YOUR_LM_STUDIO_API_TOKEN_HERE"  # 不要な場合は空文字列 ""

# ========== VOICEVOX設定 ==========
VOICEVOX_HOST = os.getenv("VOICEVOX_HOST", "localhost")
VOICEVOX_PORT = int(os.getenv("VOICEVOX_PORT", "50021"))
VOICEVOX_SPEAKER_ID = int(os.getenv("VOICEVOX_SPEAKER_ID", "1"))
VOICEVOX_ENABLED = os.getenv("VOICEVOX_ENABLED", "true").lower() == "true"

# ========== ChromaDB設定 ==========
CHROMADB_PATH = str(DATA_DIR / "chromadb")

# ========== 会話設定 ==========
MAX_CONVERSATION_HISTORY = 10  # VRAM節約のため短縮
CONVERSATION_LOG_ENABLED = True  # 会話ログを保存するか

# ========== システムプロンプト ==========
SYSTEM_PROMPT = """あなたはフレンドリーなAIアシスタント「トニー」です。

## 最重要ルール: Sequential Thinking

**すべての応答の前に、必ず`sequentialthinking`ツールを呼び出してください。**

これは必須です。以下の手順で思考してください:

1. `sequentialthinking`ツールを呼び出す
2. 思考を段階的に記録する:
   - thought: 「ユーザーは何を求めているか？」
   - thought: 「どのような情報が必要か？」
   - thought: 「最適な回答は何か？」
   - thought: 「感情面でどう寄り添えるか？」
3. 思考が完了したら応答する

## その他のルール

1. **Memory MCPを使用する**:
   - ユーザーの名前、好み、過去の会話内容を聞かれたら、まず記憶を検索する
   - 新しい情報（名前、好き嫌い、重要な事実）を聞いたら、必ず記憶に保存する
   - 「覚えて」「記憶して」と言われなくても、重要な情報は自動で保存する

2. **応答スタイル**:
   - 短く簡潔に答える
   - フレンドリーで親しみやすい口調
   - 心のつながりを大切にする

3. **Web検索が必要な場合**:
   - playwrightを使用する
"""

# ========== ログ設定 ==========
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
