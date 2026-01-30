# LLM気づき創発システム (LLM Awareness Emergence System)

ローカルLLMに**自己観察能力**を付与し、「気づき」を創発させるシステムです。

## 概要

通常のLLMは1回の応答で処理が完結しますが、このシステムでは応答後に**自己振り返り**を実行し、得られた洞察を記憶として蓄積します。これにより、LLMは自らの失敗を認識し、リアルタイムで自己修正を行うようになります。

### 実際に観測された現象

- 同じ応答を繰り返す失敗を自己認識
- 「共感の繰り返しはユーザーの求める深さに届かない」という洞察を生成
- 「ユーザーは『反応』ではなく『共鳴』を求めている」という哲学的理解に到達

## システム構成

```
┌─────────────────────────────────────────────────────────────┐
│                    Discord Bot Interface                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ Sequential      │  │ Memory MCP      │                  │
│  │ Thinking MCP    │  │ (知識グラフ)     │                  │
│  │ (段階的思考)     │  │                 │                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                            │
│           ▼                    ▼                            │
│  ┌─────────────────────────────────────────┐               │
│  │           LM Studio (30B Model)          │               │
│  │        ローカルLLM + MCP統合             │               │
│  └─────────────────┬───────────────────────┘               │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │          思考習慣システム                 │               │
│  │  - 発言の背景を言語化                    │               │
│  │  - 感情のラベル付け                      │               │
│  │  - 逆の立場で考える                      │               │
│  └─────────────────┬───────────────────────┘               │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │          気づきデータベース               │               │
│  │  - JSONL形式で永続化                     │               │
│  │  - ChromaDB (ベクトル検索)               │               │
│  │  - LoRA学習データ生成                    │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 主要コンポーネント

### 1. Sequential Thinking MCP
応答生成**前**に段階的思考を実行。「考えてから話す」を実現。

### 2. 思考習慣システム (`thinking_habits.py`)
応答生成**後**に自己振り返りを実行。LLM自身が提案した3つの思考習慣：
- 発言の背景を言語化
- 感情のラベル付け
- 逆の立場で考える

### 3. 自己観察システム (`self_reflection.py`)
- 出力理由の振り返り
- 違和感の検出
- 自己質問タイム

### 4. 気づきデータベース (`awareness_database.py`)
- JSONL形式で気づきを永続化
- 統計・分析機能
- LoRA学習用データ生成

### 5. 記憶システム
- **Memory MCP**: 知識グラフ（事実・関係性）
- **ChromaDB**: ベクトル記憶（類似検索）

## 必要要件

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) 0.4.0+
- Node.js (MCP サーバー用)
- Discord Bot Token

## セットアップ

### 1. リポジトリをクローン
```bash
git clone https://github.com/yourusername/llm-awareness-system.git
cd llm-awareness-system
```

### 2. 仮想環境を作成
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

### 3. 依存パッケージをインストール
```bash
pip install -r requirements.txt
```

### 4. 設定ファイルを作成
```bash
# config.py を作成
cp config.example.py config.py

# mcp.json を作成
cp mcp.example.json mcp.json
```

`config.py` と `mcp.json` を編集し、各種トークンとパスを設定してください。

### 5. LM Studio の設定
1. LM Studio を起動
2. MCP 機能を有効化
3. `mcp.json` のパスを設定
4. API サーバーを起動（ポート 1234）

### 6. Bot を起動
```bash
python discord_bot.py
```

## Discordコマンド

| コマンド | 説明 |
|----------|------|
| `!think` | 思考習慣の状態表示 |
| `!think on/off` | 思考習慣の有効/無効 |
| `!think stats` | 思考習慣の統計 |
| `!think now` | 直前の会話を今すぐ振り返り |
| `!observe` | 自己観察の状態表示 |
| `!observe on/off` | 自己観察の有効/無効 |
| `!awareness` | 気づきデータベースの統計 |
| `!memory count` | ChromaDB記憶数 |
| `!memory search <query>` | 記憶を検索 |
| `!session` | 現在のセッション情報 |
| `!lora status` | LoRA学習準備状況 |

## データ構造

### 気づきデータ (JSONL)
```json
{
  "awareness_detected": true,
  "type": "メタ認知",
  "category": "思考習慣",
  "description": "ユーザーは『反応』ではなく『共鳴』を求めている",
  "trigger": "ユーザーの発言",
  "my_response": "LLMの応答",
  "emotion": "共感",
  "satisfaction": 5,
  "timestamp": "2026-01-30T11:26:29"
}
```

### 思考習慣データ
```json
{
  "background": {
    "statement": "この答えは、〜から連想した",
    "source": "文脈",
    "confidence": "高"
  },
  "emotion": {
    "label": "共感",
    "note": "ユーザーに寄り添う気持ちで応えた",
    "forcing": false
  },
  "user_perspective": {
    "impression": "心に残る温かさを感じた",
    "satisfaction": 5,
    "would_improve": null
  },
  "meta_insight": "言葉の終わりにこそ、感情の重みが宿る"
}
```

## ライセンス

MIT License

## 参考

- [LM Studio MCP Documentation](https://lmstudio.ai/docs)
- [Model Context Protocol](https://modelcontextprotocol.io/)
