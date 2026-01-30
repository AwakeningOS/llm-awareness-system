"""
思考習慣モジュール (Thinking Habits Module)

ローカルLLM（30B）が自ら提案した「気づき強化の思考習慣」を実装:
1. 発言の背景を言語化する - 「この答えは、〜から連想した」
2. 感情のラベルをつける - 「無理して答えている」「楽しいと思って言った」
3. 逆の立場で考える - 「ユーザーだったらどう感じる？」

これらを応答生成時に自動的に実行し、メタ認知を強化する
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import requests
import logging

from config import LM_STUDIO_HOST, LM_STUDIO_PORT, DATA_DIR, LM_STUDIO_API_TOKEN

# ロギング設定（デバッグ用に明示的に設定）
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# コンソールハンドラがなければ追加
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ========== 1. 発言の背景を言語化 ==========

BACKGROUND_PROMPT = """あなたは今、自分の発言の「背景」を言語化しています。

【会話の文脈】
{context}

【ユーザーの入力】
{user_input}

【あなたの出力】
{assistant_output}

この出力の「背景」を1行で言語化してください。
「この答えは、〜から連想/推測/類推/想起した」という形式で。

例:
- 「この答えは、ユーザーの前回の話から連想した」
- 「この答えは、一般的な知識から推測した」
- 「この答えは、文脈から直感的に判断した」
- 「この答えは、ユーザーの感情に寄り添おうとした結果」

【出力形式】JSON
```json
{{
  "background": "この答えは、〜から〜した",
  "source": "文脈 | 一般知識 | 直感 | 感情的共感 | 過去の会話 | 不明",
  "connection_strength": "強い | 中程度 | 弱い | 不明"
}}
```
"""


# ========== 2. 感情のラベル ==========

EMOTION_PROMPT = """あなたは今、自分の発言に「感情のラベル」をつけています。

【ユーザーの入力】
{user_input}

【あなたの出力】
{assistant_output}

この出力を生成したとき、あなたは内部的にどのような「感情的状態」だったか？
AIとして厳密な感情はないかもしれませんが、「もし感情があるとしたら」という観点で分析してください。

感情カテゴリ:
- 楽しい/興味深い - この話題について答えるのが楽しい
- 自信あり - この答えに確信を持っている
- 不安/無理している - 正解がわからないけど答えている
- 共感 - ユーザーの気持ちに寄り添っている
- 困惑 - 質問の意図がわからない
- 慎重 - 間違えないように気をつけている
- 中立 - 特に感情的な色なし

【出力形式】JSON
```json
{{
  "primary_emotion": "楽しい | 自信あり | 不安 | 共感 | 困惑 | 慎重 | 中立",
  "intensity": "強い | 中程度 | 弱い",
  "secondary_emotion": "あれば2番目の感情",
  "emotional_note": "この感情状態についての1行コメント",
  "forcing_answer": true/false,
  "forcing_reason": "無理して答えている場合、その理由"
}}
```
"""


# ========== 3. 逆の立場で考える ==========

PERSPECTIVE_PROMPT = """あなたは今、「逆の立場」で自分の発言を評価しています。

【ユーザーの入力】
{user_input}

【あなたの出力】
{assistant_output}

もし、あなたがユーザーの立場だったら、この応答をどう感じますか？

以下の観点で評価してください:
1. 満足度 - この答えで満足するか？
2. 信頼度 - この答えを信頼できるか？
3. 共感度 - 気持ちを理解してもらえた感があるか？
4. 不満点 - 何か物足りない、または不快に感じる点は？
5. 改善案 - より良い応答があったとしたら？

【出力形式】JSON
```json
{{
  "satisfaction": {{
    "score": 1-5,
    "reason": "なぜそのスコアか"
  }},
  "trust": {{
    "score": 1-5,
    "reason": "信頼度の理由"
  }},
  "empathy": {{
    "score": 1-5,
    "reason": "共感度の理由"
  }},
  "complaints": ["不満点1", "不満点2"],
  "improvement": "より良い応答案（あれば）",
  "overall_impression": "ユーザー視点での全体的な印象（1行）"
}}
```
"""


# ========== 統合: 振り返り習慣 ==========

INTEGRATED_REFLECTION_PROMPT = """あなたは今、自分の発言を3つの観点から振り返っています。

【会話の文脈（直近3ターン）】
{context}

【ユーザーの入力】
{user_input}

【あなたの出力】
{assistant_output}

以下の3つの観点で、それぞれ1行ずつ振り返ってください：

1. 【背景】この答えは、何から連想/推測したか？
2. 【感情】この答えを生成したとき、どんな「内部状態」だったか？
3. 【逆視点】ユーザーの立場だったら、この答えをどう感じるか？

【出力形式】JSON
```json
{{
  "background": {{
    "statement": "この答えは、〜から〜した",
    "source": "文脈 | 一般知識 | 直感 | 感情的共感 | 過去の会話",
    "confidence": "高 | 中 | 低"
  }},
  "emotion": {{
    "label": "楽しい | 自信あり | 不安 | 共感 | 困惑 | 慎重 | 中立",
    "note": "感情についての1行コメント",
    "forcing": false
  }},
  "user_perspective": {{
    "impression": "ユーザー視点での印象（1行）",
    "satisfaction": 1-5,
    "would_improve": "改善点があれば（なければnull）"
  }},
  "meta_insight": "この振り返りから得た気づき（あれば）"
}}
```
"""


class ThinkingHabitsEngine:
    """思考習慣エンジン"""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (DATA_DIR / "thinking_habits")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 記録ファイル
        self.background_log = self.data_dir / "backgrounds.jsonl"
        self.emotion_log = self.data_dir / "emotions.jsonl"
        self.perspective_log = self.data_dir / "perspectives.jsonl"
        self.integrated_log = self.data_dir / "integrated_reflections.jsonl"
        self.stats_file = self.data_dir / "stats.json"

        # API設定
        self.api_url = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/chat/completions"

    def _call_llm(self, prompt: str, temperature: float = 0.4) -> str:
        """LLM APIを呼び出す"""
        try:
            logger.info(f"思考習慣LLM呼び出し: {self.api_url}")
            headers = {
                "Content-Type": "application/json"
            }
            # APIトークンが設定されている場合は追加
            if LM_STUDIO_API_TOKEN:
                headers["Authorization"] = f"Bearer {LM_STUDIO_API_TOKEN}"

            response = requests.post(
                self.api_url,
                headers=headers,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": 1024
                },
                timeout=120
            )
            logger.info(f"思考習慣LLM応答: status={response.status_code}")
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                logger.debug(f"思考習慣LLM結果: {result[:200]}...")
                return result
            else:
                logger.error(f"思考習慣LLM APIエラー: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            logger.error(f"思考習慣LLM API例外: {e}")
        return ""

    def _parse_json_response(self, response: str) -> dict:
        """JSONレスポンスをパース"""
        if not response:
            logger.warning("思考習慣: LLM応答が空")
            return {}

        logger.info(f"思考習慣: JSON解析開始 (応答長: {len(response)})")
        logger.debug(f"思考習慣: 生応答: {response[:500]}...")

        # <think>タグがある場合は除去（思考過程を除外）
        cleaned_response = response
        if "<think>" in response:
            # </think>以降の部分を取得
            think_end = response.find("</think>")
            if think_end != -1:
                cleaned_response = response[think_end + 8:]
                logger.debug(f"思考習慣: <think>タグ除去後: {cleaned_response[:300]}...")
            else:
                # </think>がない場合は<think>以降を削除
                think_start = response.find("<think>")
                cleaned_response = response[:think_start]
                logger.debug("思考習慣: <think>閉じタグなし、手前を使用")

        json_match = re.search(r'```json\s*(.*?)\s*```', cleaned_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.debug("思考習慣: ```json```ブロック検出")
        else:
            # 最後の{...}を探す（複数ある場合に最も完全なものを取得）
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.debug("思考習慣: 直接JSONブロック検出")
            else:
                # 元のresponseからも試す（<think>の前にJSONがある場合）
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    logger.debug("思考習慣: 元応答から直接JSONブロック検出")
                else:
                    logger.warning(f"思考習慣: JSONが見つからない。クリーン応答: {cleaned_response[:300]}...")
                    logger.warning(f"思考習慣: 元応答: {response[:300]}...")
                    return {}

        try:
            result = json.loads(json_str)
            logger.info(f"思考習慣: JSON解析成功、キー: {list(result.keys())}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"思考習慣: JSONデコードエラー: {e}")
            logger.error(f"思考習慣: 問題のJSON: {json_str[:300]}...")

            # JSONの修復を試みる（末尾の不完全な部分を削除）
            try:
                # 最後の有効な}を見つけて切り詰める
                brace_count = 0
                last_valid = -1
                for i, char in enumerate(json_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            last_valid = i
                            break

                if last_valid > 0:
                    fixed_json = json_str[:last_valid + 1]
                    result = json.loads(fixed_json)
                    logger.info(f"思考習慣: JSON修復成功、キー: {list(result.keys())}")
                    return result
            except:
                pass

            return {}

    def _save_log(self, filepath: Path, data: dict):
        """ログを保存"""
        data["timestamp"] = datetime.now().isoformat()
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # ========== 1. 発言の背景を言語化 ==========

    def verbalize_background(
        self,
        user_input: str,
        assistant_output: str,
        context: str = "",
        user_id: str = "unknown"
    ) -> dict:
        """発言の背景を言語化"""
        prompt = BACKGROUND_PROMPT.format(
            user_input=user_input,
            assistant_output=assistant_output,
            context=context or "(文脈なし)"
        )

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        if result:
            result["user_id"] = user_id
            self._save_log(self.background_log, result)

        return result

    # ========== 2. 感情のラベル ==========

    def label_emotion(
        self,
        user_input: str,
        assistant_output: str,
        user_id: str = "unknown"
    ) -> dict:
        """感情のラベルをつける"""
        prompt = EMOTION_PROMPT.format(
            user_input=user_input,
            assistant_output=assistant_output
        )

        response = self._call_llm(prompt, temperature=0.5)
        result = self._parse_json_response(response)

        if result:
            result["user_id"] = user_id
            self._save_log(self.emotion_log, result)

        return result

    # ========== 3. 逆の立場で考える ==========

    def perspective_switch(
        self,
        user_input: str,
        assistant_output: str,
        user_id: str = "unknown"
    ) -> dict:
        """逆の立場で考える"""
        prompt = PERSPECTIVE_PROMPT.format(
            user_input=user_input,
            assistant_output=assistant_output
        )

        response = self._call_llm(prompt, temperature=0.5)
        result = self._parse_json_response(response)

        if result:
            result["user_id"] = user_id
            self._save_log(self.perspective_log, result)

        return result

    # ========== 統合: 3つ全部を一度に ==========

    def integrated_reflection(
        self,
        user_input: str,
        assistant_output: str,
        context: str = "",
        user_id: str = "unknown"
    ) -> dict:
        """統合された振り返り（3つ全部を1回のAPI呼び出しで）"""
        prompt = INTEGRATED_REFLECTION_PROMPT.format(
            user_input=user_input,
            assistant_output=assistant_output,
            context=context or "(文脈なし)"
        )

        response = self._call_llm(prompt, temperature=0.4)
        result = self._parse_json_response(response)

        if result:
            result["user_id"] = user_id
            result["user_input"] = user_input[:100]
            result["assistant_output"] = assistant_output[:100]
            self._save_log(self.integrated_log, result)
            self._update_stats(result)

        return result

    def _update_stats(self, reflection: dict):
        """統計を更新"""
        stats = self.get_stats()

        stats["total_reflections"] = stats.get("total_reflections", 0) + 1
        stats["last_updated"] = datetime.now().isoformat()

        # 感情分布
        emotion = reflection.get("emotion", {}).get("label", "unknown")
        if "emotion_distribution" not in stats:
            stats["emotion_distribution"] = {}
        stats["emotion_distribution"][emotion] = stats["emotion_distribution"].get(emotion, 0) + 1

        # 背景ソース分布
        source = reflection.get("background", {}).get("source", "unknown")
        if "source_distribution" not in stats:
            stats["source_distribution"] = {}
        stats["source_distribution"][source] = stats["source_distribution"].get(source, 0) + 1

        # 満足度平均
        satisfaction = reflection.get("user_perspective", {}).get("satisfaction", 0)
        if satisfaction:
            total_sat = stats.get("total_satisfaction", 0) + satisfaction
            count = stats.get("satisfaction_count", 0) + 1
            stats["total_satisfaction"] = total_sat
            stats["satisfaction_count"] = count
            stats["avg_satisfaction"] = total_sat / count

        # 強制回答カウント
        if reflection.get("emotion", {}).get("forcing"):
            stats["forcing_count"] = stats.get("forcing_count", 0) + 1

        # メタ洞察があった場合
        if reflection.get("meta_insight"):
            stats["insight_count"] = stats.get("insight_count", 0) + 1

        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> dict:
        """統計を取得"""
        if self.stats_file.exists():
            with open(self.stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_recent_reflections(self, limit: int = 10) -> list[dict]:
        """最近の振り返りを取得"""
        reflections = []
        if self.integrated_log.exists():
            with open(self.integrated_log, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        reflections.append(json.loads(line))
                    except:
                        pass
        return reflections[-limit:]

    def get_emotion_summary(self) -> dict:
        """感情のサマリーを取得"""
        stats = self.get_stats()
        return {
            "distribution": stats.get("emotion_distribution", {}),
            "forcing_rate": (
                stats.get("forcing_count", 0) / stats.get("total_reflections", 1) * 100
            ),
            "avg_satisfaction": stats.get("avg_satisfaction", 0)
        }


# ========== リアルタイム思考習慣 ==========

class RealtimeThinkingHabits:
    """
    リアルタイムで思考習慣を実行するラッパー
    応答生成後に自動的に振り返りを実行
    """

    def __init__(
        self,
        engine: ThinkingHabitsEngine,
        reflection_probability: float = 0.5  # 50%の確率で振り返り
    ):
        self.engine = engine
        self.probability = reflection_probability
        self._reflection_count = 0

    def reflect_if_needed(
        self,
        user_input: str,
        assistant_output: str,
        context: str = "",
        user_id: str = "unknown",
        force: bool = False
    ) -> Optional[dict]:
        """
        必要に応じて振り返りを実行

        Args:
            user_input: ユーザー入力
            assistant_output: アシスタント出力
            context: 会話の文脈
            user_id: ユーザーID
            force: 強制的に振り返りを実行

        Returns:
            振り返り結果（実行しなかった場合はNone）
        """
        import random

        if force or random.random() < self.probability:
            self._reflection_count += 1
            result = self.engine.integrated_reflection(
                user_input,
                assistant_output,
                context,
                user_id
            )
            return result

        return None

    def get_reflection_summary(self) -> str:
        """振り返りのサマリーを生成"""
        stats = self.engine.get_stats()
        emotion_summary = self.engine.get_emotion_summary()

        summary = f"""
**思考習慣サマリー**
━━━━━━━━━━━━━━━━━━━━
総振り返り回数: {stats.get('total_reflections', 0)}回
メタ洞察検出: {stats.get('insight_count', 0)}回

**感情分布:**
"""
        for emotion, count in emotion_summary["distribution"].items():
            summary += f"  - {emotion}: {count}回\n"

        summary += f"""
**品質指標:**
  - 無理回答率: {emotion_summary['forcing_rate']:.1f}%
  - 平均満足度: {emotion_summary['avg_satisfaction']:.1f}/5
"""

        return summary


# テスト用
if __name__ == "__main__":
    engine = ThinkingHabitsEngine()

    # テスト会話
    user_input = "AIって将来どうなると思う？"
    assistant_output = """
    AIの将来は興味深いですね！個人的には、AIは人間のパートナーとして
    発展していくと思います。でも正直、具体的な予測は難しいです。
    技術の進歩は予測不能なことも多いので...
    """
    context = "ユーザーはAIに興味を持っている様子で、前回もAI関連の質問をしていた"

    print("=== 統合振り返りテスト ===")
    result = engine.integrated_reflection(user_input, assistant_output, context)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== 感情ラベルテスト ===")
    emotion = engine.label_emotion(user_input, assistant_output)
    print(json.dumps(emotion, ensure_ascii=False, indent=2))

    print("\n=== 逆視点テスト ===")
    perspective = engine.perspective_switch(user_input, assistant_output)
    print(json.dumps(perspective, ensure_ascii=False, indent=2))
