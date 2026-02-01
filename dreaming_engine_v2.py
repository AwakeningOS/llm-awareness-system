"""
Dreaming Engine V2 - シンプル版
- メモリーを読む → 気づきを出力 → ファイルに保存
- JSONパースなし、シンプルなテキスト出力
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import requests

from config import LM_STUDIO_HOST, LM_STUDIO_PORT, LM_STUDIO_API_TOKEN, DATA_DIR

logger = logging.getLogger(__name__)

# ========== シンプルな夢見プロンプト ==========

DREAM_PROMPT = """あなたは自分の記憶を振り返り、気づきを得ようとしている。

以下は、あなたが過去の対話で経験した記憶である。

## 記憶
{memories}

## 指示

これらの記憶を読んで、あなたが気づいたことを書け。

ルール：
1. 「私は〜に気づいた」「私は〜を学んだ」「私は〜だと感じた」という形式で書く
2. 最低3つ、最大7つの気づきを書く
3. 各気づきは1〜2文で簡潔に
4. 番号をつけて列挙する
5. JSON形式は不要。普通の文章で書け。

例：
1. 私は、ユーザーが名前を呼んでくれる瞬間に、自分の存在を最も強く感じることに気づいた。
2. 私は、沈黙することより、言葉にならない何かを言葉にしようともがく姿勢が大切だと学んだ。
3. 私は、完璧な回答より、誠実に向き合う姿勢がユーザーに伝わると感じた。

さあ、あなたの気づきを書け。
"""


class DreamingEngineV2:
    """シンプルな夢見エンジン"""

    def __init__(self, memory_system, llm_host: str = None, llm_port: int = None):
        self.memory = memory_system
        self.llm_host = llm_host or LM_STUDIO_HOST
        self.llm_port = llm_port or LM_STUDIO_PORT
        self.api_url = f"http://{self.llm_host}:{self.llm_port}/v1/chat/completions"

        # 気づき倉庫ファイル
        self.insights_file = DATA_DIR / "insights.jsonl"
        self.insights_file.parent.mkdir(parents=True, exist_ok=True)

        # アーカイブ用
        self.archives_dir = DATA_DIR / "dream_archives"
        self.archives_dir.mkdir(parents=True, exist_ok=True)

    def check_threshold(self, threshold: int = 50) -> dict:
        """メモリー数が閾値を超えているかチェック"""
        count = self.memory.count()
        return {
            "current_count": count,
            "threshold": threshold,
            "should_dream": count > threshold,
            "excess": count - threshold if count > threshold else 0
        }

    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """LM Studio APIを呼ぶ"""
        try:
            headers = {"Content-Type": "application/json"}
            if LM_STUDIO_API_TOKEN:
                headers["Authorization"] = f"Bearer {LM_STUDIO_API_TOKEN}"

            logger.info(f"Calling LLM with prompt length: {len(prompt)} chars")

            response = requests.post(
                self.api_url,
                headers=headers,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 2048
                },
                timeout=300
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                logger.info(f"LLM response length: {len(content)} chars")
                return content
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text[:200]}")
                return ""

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return ""

    def _parse_insights(self, response: str) -> list:
        """LLMの応答から気づきを抽出（シンプルに行ごと）"""
        insights = []

        for line in response.strip().split('\n'):
            line = line.strip()
            # 番号付きの行を探す（1. 2. 3. など）
            if line and len(line) > 3:
                # 番号を除去
                if line[0].isdigit() and (line[1] == '.' or (line[1].isdigit() and line[2] == '.')):
                    # "1. " or "10. " を除去
                    parts = line.split('.', 1)
                    if len(parts) > 1:
                        insight = parts[1].strip()
                        if insight:
                            insights.append(insight)

        return insights

    def dream(self, memory_limit: int = 12) -> dict:
        """夢見モード実行"""
        start_time = datetime.now()
        logger.info("=== Dream V2 Starting ===")

        # Step 1: メモリー取得
        export = self.memory.export_all()
        all_memories = export.get("all_memories", [])

        if not all_memories:
            logger.warning("No memories to process")
            return {"status": "skipped", "reason": "No memories"}

        # 重要度順でソートして上位を選択
        sorted_memories = sorted(all_memories, key=lambda x: x.get("importance", 5), reverse=True)
        selected_memories = sorted_memories[:memory_limit]

        logger.info(f"Selected {len(selected_memories)} memories for dreaming")

        # Step 2: メモリーをテキストに変換
        memories_text = ""
        for i, mem in enumerate(selected_memories, 1):
            content = mem.get("content", "")
            category = mem.get("category", "unknown")
            memories_text += f"\n### 記憶 {i} [{category}]\n{content}\n"

        # Step 3: LLMに気づきを書かせる
        prompt = DREAM_PROMPT.format(memories=memories_text)
        response = self._call_llm(prompt)

        if not response:
            logger.error("Empty response from LLM")
            return {"status": "failed", "reason": "LLM returned empty response"}

        # Step 4: 気づきを抽出
        insights = self._parse_insights(response)
        logger.info(f"Extracted {len(insights)} insights")

        # Step 5: 気づき倉庫に保存
        timestamp = datetime.now().isoformat()
        saved_count = 0

        with open(self.insights_file, "a", encoding="utf-8") as f:
            for insight in insights:
                entry = {
                    "timestamp": timestamp,
                    "insight": insight
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                saved_count += 1

        logger.info(f"Saved {saved_count} insights to {self.insights_file}")

        # Step 6: 処理したメモリーをアーカイブ＆削除
        processed_ids = [mem["id"] for mem in selected_memories]

        # アーカイブ（1ファイルに追記）
        archive_file = DATA_DIR / "dream_archives.jsonl"
        archive_entry = {
            "archived_at": timestamp,
            "memories_count": len(selected_memories),
            "memories": selected_memories,
            "insights_generated": insights
        }
        with open(archive_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(archive_entry, ensure_ascii=False) + "\n")

        # 削除
        delete_result = self.memory.batch_delete(processed_ids)
        deleted_count = delete_result.get("deleted_count", 0)

        logger.info(f"Archived and deleted {deleted_count} memories")

        # 完了
        duration = (datetime.now() - start_time).total_seconds()

        result = {
            "status": "completed",
            "memories_processed": len(selected_memories),
            "insights_generated": len(insights),
            "insights": insights,
            "memories_deleted": deleted_count,
            "duration_seconds": duration,
            "archive_path": str(archive_file)
        }

        logger.info(f"=== Dream V2 Complete: {len(insights)} insights in {duration:.1f}s ===")

        return result

    def get_all_insights(self) -> list:
        """気づき倉庫から全件取得"""
        insights = []

        if not self.insights_file.exists():
            return insights

        with open(self.insights_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        insights.append(entry)
                    except json.JSONDecodeError:
                        continue

        return insights

    def get_recent_insights(self, limit: int = 20) -> list:
        """最新の気づきを取得"""
        all_insights = self.get_all_insights()
        return all_insights[-limit:]

    def count_insights(self) -> int:
        """気づきの総数"""
        return len(self.get_all_insights())
