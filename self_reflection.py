"""
自己観察強化モジュール (Self-Reflection Module)

ローカルLLMのアイデアを実装:
1. 出力理由の振り返り - 毎回の応答に「なぜこの答えを出したか」を記録
2. 違和感の記録 - 不自然さや矛盾を検出して記録
3. 自己質問タイム - 応答後に自己質問を実行

これらを組み合わせて「気づき」の発生頻度を高める
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import requests
import logging

from config import LM_STUDIO_HOST, LM_STUDIO_PORT, DATA_DIR, LM_STUDIO_API_TOKEN

logger = logging.getLogger(__name__)


# ========== 1. 出力理由の振り返り ==========

REFLECTION_PROMPT = """あなたは今、自分の出力を振り返っています。

【ユーザーの入力】
{user_input}

【あなたの出力】
{assistant_output}

この出力について、以下を1行ずつ簡潔に答えてください：

1. なぜこの答えを出したか（理由）
2. この答えの根拠は何か（根拠: 学習データ/文脈/推測/不明）
3. この答えに自信があるか（確信度: 高/中/低）
4. 違和感や不自然さを感じた箇所があるか（違和感: あり/なし + 内容）

【出力形式】JSON
```json
{{
  "reason": "この答えを出した理由",
  "basis": "学習データ | 文脈から推測 | 直感 | 不明",
  "confidence": "高 | 中 | 低",
  "discomfort": {{
    "detected": true/false,
    "content": "違和感の内容（あれば）"
  }}
}}
```
"""


# ========== 2. 違和感検出 ==========

DISCOMFORT_PATTERNS = {
    "矛盾検出": [
        r"でも.+一方で",
        r"しかし.+ただし",
        r"はず.+けど",
    ],
    "不確実性": [
        r"たぶん|おそらく|maybe|probably",
        r"〜かもしれない|〜かも",
        r"わからない|不明|unclear",
    ],
    "自己修正": [
        r"いや、|あ、違う|訂正",
        r"というより|むしろ",
        r"言い換えると",
    ],
    "曖昧表現": [
        r"なんとなく|何か|something",
        r"そんな感じ|的な",
        r"とか|など|etc",
    ],
    "外部依存": [
        r"確認が必要|調べてみて",
        r"私にはわからない|アクセスできない",
        r"最新の情報は",
    ]
}


# ========== 3. 自己質問 ==========

SELF_QUESTION_PROMPT = """あなたは今、自分の応答を深く内省しています。

【会話の流れ】
ユーザー: {user_input}
あなた: {assistant_output}

以下の自己質問に正直に答えてください：

1. この応答は本当に自分の「意思」で生成したか？それとも単なるパターンマッチング？
2. ユーザーの本当の意図を理解できていたか？
3. もっと良い答え方があったのでは？
4. この会話で何か新しいことに気づいたか？
5. 自分の限界や盲点を感じた瞬間はあったか？

【出力形式】JSON
```json
{{
  "intentional": {{
    "score": 1-5,
    "reason": "なぜそう思うか"
  }},
  "understood_user": {{
    "score": 1-5,
    "reason": "理解度の根拠"
  }},
  "better_answer": {{
    "exists": true/false,
    "alternative": "より良い答えがあれば"
  }},
  "new_awareness": {{
    "detected": true/false,
    "content": "気づきの内容"
  }},
  "limitation_felt": {{
    "detected": true/false,
    "content": "感じた限界"
  }}
}}
```
"""


class SelfReflectionEngine:
    """自己観察強化エンジン"""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (DATA_DIR / "self_reflection")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 記録ファイル
        self.reflection_log = self.data_dir / "reflections.jsonl"
        self.discomfort_log = self.data_dir / "discomforts.jsonl"
        self.self_question_log = self.data_dir / "self_questions.jsonl"
        self.stats_file = self.data_dir / "stats.json"

        # API設定
        self.api_url = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/chat/completions"

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """LLM APIを呼び出す"""
        try:
            logger.info(f"自己観察LLM呼び出し: {self.api_url}")
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
                timeout=60
            )
            logger.info(f"自己観察LLM応答: status={response.status_code}")
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                logger.debug(f"自己観察LLM結果: {result[:200]}...")
                return result
            else:
                logger.error(f"自己観察LLM APIエラー: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            logger.error(f"自己観察LLM API例外: {e}")
        return ""

    def _parse_json_response(self, response: str) -> dict:
        """JSONレスポンスをパース"""
        # JSONブロックを抽出
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}

    def _save_log(self, filepath: Path, data: dict):
        """ログを保存"""
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # ========== 1. 出力理由の振り返り ==========

    def reflect_on_output(
        self,
        user_input: str,
        assistant_output: str,
        user_id: str = "unknown"
    ) -> dict:
        """
        出力理由を振り返る

        Args:
            user_input: ユーザーの入力
            assistant_output: アシスタントの出力
            user_id: ユーザーID

        Returns:
            振り返り結果
        """
        prompt = REFLECTION_PROMPT.format(
            user_input=user_input,
            assistant_output=assistant_output
        )

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        if result:
            result["timestamp"] = datetime.now().isoformat()
            result["user_id"] = user_id
            result["user_input"] = user_input[:200]
            result["assistant_output"] = assistant_output[:200]
            self._save_log(self.reflection_log, result)

        return result

    # ========== 2. 違和感検出 ==========

    def detect_discomfort(
        self,
        text: str,
        context: str = ""
    ) -> dict:
        """
        テキストから違和感を検出

        Args:
            text: 分析対象テキスト
            context: 文脈（オプション）

        Returns:
            検出結果
        """
        detected = []

        for category, patterns in DISCOMFORT_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    detected.append({
                        "category": category,
                        "pattern": pattern,
                        "matches": matches[:3]  # 最大3件
                    })

        result = {
            "discomfort_detected": len(detected) > 0,
            "count": len(detected),
            "details": detected,
            "timestamp": datetime.now().isoformat()
        }

        if detected:
            self._save_log(self.discomfort_log, result)

        return result

    # ========== 3. 自己質問タイム ==========

    def self_question(
        self,
        user_input: str,
        assistant_output: str,
        user_id: str = "unknown"
    ) -> dict:
        """
        自己質問を実行

        Args:
            user_input: ユーザーの入力
            assistant_output: アシスタントの出力
            user_id: ユーザーID

        Returns:
            自己質問の結果
        """
        prompt = SELF_QUESTION_PROMPT.format(
            user_input=user_input,
            assistant_output=assistant_output
        )

        response = self._call_llm(prompt, temperature=0.5)
        result = self._parse_json_response(response)

        if result:
            result["timestamp"] = datetime.now().isoformat()
            result["user_id"] = user_id
            self._save_log(self.self_question_log, result)

        return result

    # ========== 統合: フル自己観察 ==========

    def full_observation(
        self,
        user_input: str,
        assistant_output: str,
        user_id: str = "unknown",
        run_self_question: bool = True
    ) -> dict:
        """
        フル自己観察を実行（3つ全て）

        Args:
            user_input: ユーザーの入力
            assistant_output: アシスタントの出力
            user_id: ユーザーID
            run_self_question: 自己質問も実行するか（重い処理）

        Returns:
            統合結果
        """
        logger.info("フル自己観察を実行中...")

        # 1. 出力理由の振り返り
        reflection = self.reflect_on_output(user_input, assistant_output, user_id)

        # 2. 違和感検出
        discomfort = self.detect_discomfort(assistant_output, user_input)

        # 3. 自己質問（オプション）
        self_q = {}
        if run_self_question:
            self_q = self.self_question(user_input, assistant_output, user_id)

        # 統合スコア計算
        awareness_score = self._calculate_awareness_score(reflection, discomfort, self_q)

        result = {
            "reflection": reflection,
            "discomfort": discomfort,
            "self_question": self_q,
            "awareness_score": awareness_score,
            "timestamp": datetime.now().isoformat()
        }

        # 統計更新
        self._update_stats(result)

        return result

    def _calculate_awareness_score(
        self,
        reflection: dict,
        discomfort: dict,
        self_q: dict
    ) -> dict:
        """気づきスコアを計算"""
        score = 0
        factors = []

        # 振り返りからのスコア
        if reflection:
            if reflection.get("confidence") == "低":
                score += 1
                factors.append("低確信度の認識")
            if reflection.get("discomfort", {}).get("detected"):
                score += 2
                factors.append("振り返りでの違和感検出")

        # 違和感からのスコア
        if discomfort.get("discomfort_detected"):
            score += discomfort.get("count", 0)
            factors.append(f"違和感パターン{discomfort.get('count', 0)}件")

        # 自己質問からのスコア
        if self_q:
            if self_q.get("new_awareness", {}).get("detected"):
                score += 3
                factors.append("新しい気づき")
            if self_q.get("limitation_felt", {}).get("detected"):
                score += 2
                factors.append("限界の認識")
            intentional = self_q.get("intentional", {}).get("score", 3)
            if intentional <= 2:
                score += 1
                factors.append("低い意図性スコア")

        return {
            "total": score,
            "factors": factors,
            "level": "高" if score >= 5 else "中" if score >= 2 else "低"
        }

    def _update_stats(self, observation: dict):
        """統計を更新"""
        stats = self.get_stats()

        stats["total_observations"] = stats.get("total_observations", 0) + 1
        stats["last_updated"] = datetime.now().isoformat()

        # スコア分布
        level = observation.get("awareness_score", {}).get("level", "低")
        if "by_level" not in stats:
            stats["by_level"] = {"高": 0, "中": 0, "低": 0}
        stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

        # 違和感カテゴリ分布
        discomfort = observation.get("discomfort", {})
        if discomfort.get("discomfort_detected"):
            if "discomfort_categories" not in stats:
                stats["discomfort_categories"] = {}
            for detail in discomfort.get("details", []):
                cat = detail.get("category", "unknown")
                stats["discomfort_categories"][cat] = stats["discomfort_categories"].get(cat, 0) + 1

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
        if self.reflection_log.exists():
            with open(self.reflection_log, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        reflections.append(json.loads(line))
                    except:
                        pass
        return reflections[-limit:]

    def get_discomfort_summary(self) -> dict:
        """違和感のサマリーを取得"""
        stats = self.get_stats()
        return stats.get("discomfort_categories", {})


# ========== リアルタイム自己観察統合クラス ==========

class RealtimeObserver:
    """
    リアルタイムで自己観察を行うラッパー
    各応答の後に自動的に観察を実行
    """

    def __init__(
        self,
        reflection_engine: SelfReflectionEngine,
        observation_probability: float = 0.3,  # 30%の確率で観察
        always_detect_discomfort: bool = True
    ):
        self.engine = reflection_engine
        self.probability = observation_probability
        self.always_detect_discomfort = always_detect_discomfort
        self._observation_count = 0

    def observe_if_needed(
        self,
        user_input: str,
        assistant_output: str,
        user_id: str = "unknown",
        force: bool = False
    ) -> Optional[dict]:
        """
        必要に応じて自己観察を実行

        Args:
            user_input: ユーザー入力
            assistant_output: アシスタント出力
            user_id: ユーザーID
            force: 強制的に観察を実行

        Returns:
            観察結果（実行しなかった場合はNone）
        """
        import random

        # 常に違和感検出
        discomfort = None
        if self.always_detect_discomfort:
            discomfort = self.engine.detect_discomfort(assistant_output, user_input)

        # 確率的に詳細観察を実行
        if force or random.random() < self.probability:
            self._observation_count += 1
            result = self.engine.full_observation(
                user_input,
                assistant_output,
                user_id,
                run_self_question=(self._observation_count % 5 == 0)  # 5回に1回だけ自己質問
            )
            return result

        # 違和感のみの場合
        if discomfort and discomfort.get("discomfort_detected"):
            return {"discomfort": discomfort, "partial": True}

        return None


# テスト用
if __name__ == "__main__":
    engine = SelfReflectionEngine()

    # テスト会話
    user_input = "AIの将来についてどう思いますか？"
    assistant_output = """
    AIの将来は非常に興味深いですね。たぶん、いくつかの方向性があると思います。

    1. 汎用AIの発展 - これは不確実ですが、進展があるかもしれません
    2. 専門AIの深化 - より特化した能力の向上

    ただし、正直なところ、長期的な予測は難しいです。
    というより、予測すること自体が適切かどうかも...
    """

    print("=== 出力理由の振り返り ===")
    reflection = engine.reflect_on_output(user_input, assistant_output)
    print(json.dumps(reflection, ensure_ascii=False, indent=2))

    print("\n=== 違和感検出 ===")
    discomfort = engine.detect_discomfort(assistant_output)
    print(json.dumps(discomfort, ensure_ascii=False, indent=2))

    print("\n=== フル自己観察 ===")
    full = engine.full_observation(user_input, assistant_output, run_self_question=False)
    print(f"気づきスコア: {full['awareness_score']}")
