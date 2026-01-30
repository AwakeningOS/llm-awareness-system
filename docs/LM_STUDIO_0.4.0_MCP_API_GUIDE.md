# LM Studio 0.4.0 MCP/API 完全ガイド

このドキュメントは、LM Studio 0.4.0のMCP（Model Context Protocol）機能とAPIの使い方をまとめたものです。

---

## 目次

1. [概要](#概要)
2. [APIエンドポイント](#apiエンドポイント)
3. [MCP設定](#mcp設定)
4. [JIT（Just-in-Time）ロード](#jitjust-in-timeロード)
5. [よくあるエラーと解決策](#よくあるエラーと解決策)
6. [実装例](#実装例)

---

## 概要

LM Studio 0.4.0では、MCPサーバーとの統合が強化され、外部ツール（ファイルシステム、メモリ、Playwrightなど）をLLMから直接呼び出せるようになりました。

### 重要な変更点（0.3.x → 0.4.0）

| 項目 | 0.3.x | 0.4.0 |
|------|-------|-------|
| MCPエンドポイント | なし | `/api/v1/chat` |
| OpenAI互換 | `/v1/chat/completions` | 引き続き利用可能 |
| JITロード | なし | サポート |
| TTLパラメータ | - | `/api/v1/chat`では**非対応** |

---

## APIエンドポイント

### 1. `/api/v1/chat`（MCP対応・推奨）

MCPツールを使用する場合はこのエンドポイントを使用します。

```python
import requests

url = "http://localhost:1234/api/v1/chat"

payload = {
    "model": "qwen2.5-7b-instruct",  # モデル名
    "input": "私の名前はジョージです。覚えてください。",
    "system_prompt": "あなたはAIアシスタントです。",
    "integrations": ["memory", "filesystem"],  # 使用するMCPサーバー
    "context_length": 8192,
    "temperature": 0.7
}

# 認証（設定している場合）
headers = {
    "Authorization": "Bearer sk-lm-xxxxx"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

#### パラメータ説明

| パラメータ | 必須 | 説明 |
|-----------|------|------|
| `model` | ✅ | 使用するモデル名（LM Studioで表示される名前） |
| `input` | ✅ | ユーザーの入力メッセージ |
| `system_prompt` | ❌ | システムプロンプト |
| `integrations` | ❌ | 使用するMCPサーバー名の配列 |
| `context_length` | ❌ | コンテキスト長（デフォルト: モデル依存） |
| `temperature` | ❌ | 温度パラメータ（0.0〜2.0） |

#### ⚠️ 注意: `ttl`は非対応

`/api/v1/chat`では`ttl`パラメータは**サポートされていません**。
TTLを設定したい場合は、LM Studio GUIまたはCLIで設定してください。

```python
# ❌ エラーになる
payload = {
    "model": "qwen2.5-7b-instruct",
    "input": "Hello",
    "ttl": 300  # Unrecognized key(s) in object: 'ttl'
}

# ✅ 正しい
payload = {
    "model": "qwen2.5-7b-instruct",
    "input": "Hello"
}
```

### 2. `/v1/chat/completions`（OpenAI互換）

MCP機能を使わない場合や、OpenAI互換が必要な場合に使用します。

```python
url = "http://localhost:1234/v1/chat/completions"

payload = {
    "model": "qwen2.5-7b-instruct",
    "messages": [
        {"role": "system", "content": "あなたはAIアシスタントです。"},
        {"role": "user", "content": "こんにちは"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000,
    "ttl": 300  # こちらでは使用可能
}

response = requests.post(url, json=payload)
```

---

## MCP設定

### 設定ファイルの場所

```
Windows: C:\Users\<ユーザー名>\.lmstudio\mcp.json
Mac: ~/.lmstudio/mcp.json
Linux: ~/.lmstudio/mcp.json
```

### mcp.json の例

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "env": {
        "MEMORY_FILE_PATH": "C:/Users/youthk/Desktop/mcp_agent/data/memory.jsonl"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:/Users/youthk/Desktop/mcp_agent/workspace"
      ]
    },
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp@latest"],
      "timeout": 60000
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    }
  }
}
```

### 主要なMCPサーバー

| サーバー名 | パッケージ | 機能 |
|-----------|-----------|------|
| memory | `@modelcontextprotocol/server-memory` | 永続的な記憶（Knowledge Graph） |
| filesystem | `@modelcontextprotocol/server-filesystem` | ファイル読み書き |
| playwright | `@playwright/mcp` | Webブラウザ操作 |
| sequential-thinking | `@modelcontextprotocol/server-sequential-thinking` | 段階的思考 |

### Memory MCPのデータ形式

Memory MCPはKnowledge Graph形式でデータを保存します。

```jsonl
{"type":"entity","name":"ジョージ","entityType":"人物","observations":["名前はジョージです。","ラーメンが好き"]}
{"type":"entity","name":"プロジェクトX","entityType":"プロジェクト","observations":["Pythonで開発中","Discord Botと連携"]}
```

---

## JIT（Just-in-Time）ロード

### 重要な仕様

LM Studio 0.4.0では、**モデルを手動でロードしないでください**。

APIリクエストを受けると、指定されたモデルが自動的にロードされます（JITロード）。

```
❌ やってはいけないこと:
1. LM Studio GUIでモデルを手動ロード
2. その後APIリクエストを送信
→ 同じモデルが2つロードされてVRAMを無駄に消費

✅ 正しい使い方:
1. LM Studioを起動（モデルはロードしない）
2. APIリクエストを送信
→ 自動的にモデルがロードされる
```

### モデル名の確認方法

```bash
# ロード済みモデルの確認
curl http://localhost:1234/v1/models

# レスポンス例
{
  "data": [
    {
      "id": "qwen2.5-7b-instruct",
      "object": "model",
      ...
    }
  ]
}
```

---

## よくあるエラーと解決策

### 1. `Unrecognized key(s) in object: 'ttl'`

**原因**: `/api/v1/chat`で`ttl`パラメータを使用

**解決策**: `ttl`をペイロードから削除

```python
# ❌
payload = {"model": "...", "input": "...", "ttl": 300}

# ✅
payload = {"model": "...", "input": "..."}
```

### 2. `Model not found`

**原因**: モデル名が間違っている、またはモデルがインストールされていない

**解決策**:
- LM Studio GUIで正確なモデル名を確認
- モデル名はファイル名ではなく、LM Studioで表示される名前を使用

### 3. MCPツールが使われない

**原因**: モデルがMCPツールの使い方を理解していない

**解決策**: システムプロンプトで明示的に指示

```python
system_prompt = """あなたはAIアシスタントです。

## 重要なルール
1. **必ずMemory MCPを使用する**:
   - ユーザーの名前、好み、過去の会話内容を聞かれたら、まず記憶を検索する
   - 新しい情報を聞いたら、必ず記憶に保存する
   - 「覚えて」と言われなくても、重要な情報は自動で保存する
"""
```

### 4. VRAMが足りない / 同じモデルが複数ロードされる

**原因**: 手動でモデルをロードした後にAPIを呼び出した

**解決策**:
- LM Studioでロード済みのモデルをアンロード
- JITロードに任せる（モデルを手動でロードしない）

### 5. MCP接続エラー

**原因**: MCPサーバーが起動していない、または設定が間違っている

**解決策**:
1. `mcp.json`のパスが正しいか確認
2. Node.js/npxがインストールされているか確認
3. LM Studioを再起動

---

## 実装例

### Discord Bot with Memory MCP

```python
import requests
import json

# 設定
LM_STUDIO_MCP_URL = "http://localhost:1234/api/v1/chat"
API_TOKEN = "sk-lm-xxxxx"  # 設定している場合

# MCPサーバー設定
MCP_INTEGRATIONS = ["memory", "playwright"]

def get_auth_headers():
    return {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

async def chat_with_mcp(user_input: str, system_prompt: str, model_name: str):
    """MCPを使用してLLMと会話"""

    payload = {
        "model": model_name,
        "input": user_input,
        "system_prompt": system_prompt,
        "integrations": MCP_INTEGRATIONS,
        "context_length": 8192,
        "temperature": 0.7,
        # 注: ttlは /api/v1/chat では非対応
    }

    try:
        response = requests.post(
            LM_STUDIO_MCP_URL,
            headers=get_auth_headers(),
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
```

### Memory MCPを活用するシステムプロンプト

```python
SYSTEM_PROMPT = """あなたはフレンドリーなAIアシスタント「トニー」です。

## 重要なルール
1. **必ずMemory MCPを使用する**:
   - ユーザーの名前、好み、過去の会話内容を聞かれたら、まず記憶を検索する
   - 新しい情報（名前、好き嫌い、重要な事実）を聞いたら、必ず記憶に保存する
   - 「覚えて」「記憶して」と言われなくても、重要な情報は自動で保存する

2. **応答スタイル**:
   - 短く簡潔に答える
   - フレンドリーで親しみやすい口調

3. **Web検索が必要な場合**:
   - playwrightを使用する
"""
```

---

## 参考リンク

- [LM Studio 公式ドキュメント](https://lmstudio.ai/docs)
- [MCP 公式仕様](https://modelcontextprotocol.io/)
- [Memory MCP Server](https://github.com/modelcontextprotocol/servers/tree/main/src/memory)

---

## 更新履歴

| 日付 | 内容 |
|------|------|
| 2025-01-30 | 初版作成（LM Studio 0.4.0対応） |
