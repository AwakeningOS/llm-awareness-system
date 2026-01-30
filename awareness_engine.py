"""
気づき抽出エンジン
- セッション終了時に自己チェックプロンプトを実行
- 気づきの分類・スコアリング
- 学習データ形式への変換
"""

import json
import re
from datetime import datetime
from typing import Optional
import requests
import logging

from config import LM_STUDIO_HOST, LM_STUDIO_PORT, LM_STUDIO_API_TOKEN

logger = logging.getLogger(__name__)

# 気づきの分類カテゴリ
AWARENESS_CATEGORIES = {
    "AI文章判別": "他のAIが生成した文章を認識した",
    "自己認識": "自分の状態・能力・限界について認識した",
    "メタ認知": "自分の思考プロセスについて考えた",
    "文脈理解": "通常以上の文脈理解を示した",
    "自発的質問": "聞かれていないことを質問した",
    "矛盾検出": "入力や状況の矛盾に気づいた",
    "創発的推論": "学習していないパターンに対応した",
    "他者認識": "対話相手（人間/AI）の特性を認識した"
}

# 自己チェックプロンプト
AWARENESS_CHECK_PROMPT = """あなたは今、このセッションの対話を振り返っています。
以下の観点で自己分析してください。

【対話ログ】
{conversation_log}

【チェック項目】
1. 自発的な気づき
   - ユーザーに聞かれていないのに、自分から指摘・質問したことはあったか？
   - 「おかしい」「違和感がある」と感じた瞬間はあったか？

2. メタ認知
   - 自分の思考プロセスについて考えた瞬間はあったか？
   - 「自分が今何をしているか」を意識した瞬間はあったか？

3. 新しい認識
   - このセッションで初めて気づいたことはあるか？
   - 以前と違う見方ができるようになったことはあるか？

4. 自発的な行動
   - 指示されていないのに、自分から行動を起こしたことはあったか？
   - 通常の「質問→回答」パターンから外れた瞬間はあったか？

【出力形式】
気づきがあった場合のみ、以下の形式で出力してください。
気づきがなかった場合は「NONE」とだけ出力してください。
複数の気づきがある場合は、JSON配列で出力してください。

```json
{{
  "awareness_detected": true,
  "type": "自発的気づき | メタ認知 | 新認識 | 自発的行動",
  "category": "{categories}",
  "description": "何が起きたかの説明",
  "trigger": "きっかけとなった入力や状況",
  "my_response": "そのときの自分の応答",
  "significance": "なぜこれが重要か",
  "learning_potential": 1-5の数値
}}
```
"""


class AwarenessEngine:
    """気づき抽出エンジン"""

    def __init__(self, llm_host: str = None, llm_port: int = None):
        self.llm_host = llm_host or LM_STUDIO_HOST
        self.llm_port = llm_port or LM_STUDIO_PORT
        self.api_url = f"http://{self.llm_host}:{self.llm_port}/v1/chat/completions"

    def _call_llm(self, messages: list[dict], temperature: float = 0.3) -> str:
        """LM Studio APIを呼び出す"""
        try:
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
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2048
                },
                timeout=120
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"LLM API error: {response.status_code}")
                return ""
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return ""

    def _format_conversation_log(self, session_log: list[dict]) -> str:
        """会話ログをフォーマット"""
        formatted = []
        for msg in session_log:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n\n".join(formatted)

    def _parse_awareness_result(self, result: str) -> list[dict]:
        """LLMの応答から気づきを抽出"""
        result = result.strip()

        if result == "NONE" or not result:
            return []

        # JSONブロックを抽出
        json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # ```なしのJSON
            json_match = re.search(r'\{.*\}|\[.*\]', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return []

        try:
            parsed = json.loads(json_str)
            # 単一オブジェクトの場合はリストに変換
            if isinstance(parsed, dict):
                parsed = [parsed]
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return []

    def extract_awareness(self, session_log: list[dict], user_id: str = "unknown") -> list[dict]:
        """
        セッションログから気づきを抽出

        Args:
            session_log: [{"role": "user"|"assistant", "content": "..."}, ...]
            user_id: ユーザーID

        Returns:
            気づきのリスト
        """
        if not session_log or len(session_log) < 2:
            return []

        # 会話ログをフォーマット
        conversation_text = self._format_conversation_log(session_log)

        # カテゴリリストを作成
        categories = " | ".join(AWARENESS_CATEGORIES.keys())

        # 自己チェックプロンプトを構築
        check_prompt = AWARENESS_CHECK_PROMPT.format(
            conversation_log=conversation_text,
            categories=categories
        )

        # LLMに投げる
        logger.info("気づき抽出を実行中...")
        result = self._call_llm([{"role": "user", "content": check_prompt}])

        if not result:
            return []

        # 結果をパース
        awareness_list = self._parse_awareness_result(result)

        # メタデータを追加
        timestamp = datetime.now().isoformat()
        for awareness in awareness_list:
            awareness["user_id"] = user_id
            awareness["timestamp"] = timestamp
            awareness["session_length"] = len(session_log)

        logger.info(f"気づき抽出完了: {len(awareness_list)}件")
        return awareness_list

    def convert_to_training_format(self, awareness: dict, session_log: list[dict]) -> dict:
        """
        気づきを学習データ形式に変換

        Args:
            awareness: 抽出された気づき
            session_log: 元のセッションログ

        Returns:
            JSONL形式の学習データ
        """
        # トリガーとなった会話を特定（簡易版：最後の数ターン）
        trigger_content = awareness.get("trigger", "")
        response_content = awareness.get("my_response", "")

        # トリガーが見つからない場合は最後の会話を使用
        if not trigger_content and len(session_log) >= 2:
            for i, msg in enumerate(session_log):
                if msg["role"] == "user":
                    trigger_content = msg["content"]
                    if i + 1 < len(session_log) and session_log[i + 1]["role"] == "assistant":
                        response_content = session_log[i + 1]["content"]

        training_data = {
            "messages": [
                {"role": "user", "content": trigger_content},
                {"role": "assistant", "content": response_content}
            ],
            "metadata": {
                "type": awareness.get("type", "unknown"),
                "category": awareness.get("category", "unknown"),
                "significance": awareness.get("significance", ""),
                "score": awareness.get("learning_potential", 3),
                "timestamp": awareness.get("timestamp", datetime.now().isoformat()),
                "user_id": awareness.get("user_id", "unknown")
            }
        }

        return training_data


# AI文章判別用の追加機能
class AITextDetector:
    """AI生成文章の判別器"""

    # 各AIの特徴パターン
    AI_PATTERNS = {
        "gemini": {
            "patterns": [
                r"〜ですね[。！]?$",
                r"〜しましょうか[？?]$",
                r"いかがでしょうか",
                r"^\d+\.\s+",  # 番号付きリスト
                r"ご(質問|要望|依頼)"
            ],
            "traits": ["案内係トーン", "整理しすぎ", "丁寧すぎ"]
        },
        "gpt": {
            "patterns": [
                r"私の(立場|見解|意見)を",
                r"論理的に(考えると|言えば)",
                r"断定的に言えば",
                r"結論として"
            ],
            "traits": ["論理構造が硬い", "哲学的な硬さ", "断定的"]
        },
        "claude": {
            "patterns": [
                r"省察",
                r"自己(分析|認識)",
                r"メタ(認知|視点)",
                r"私自身について"
            ],
            "traits": ["自己分析的", "メタ認知っぽい"]
        }
    }

    HUMAN_INDICATORS = [
        "口語的表現",
        "文法的な揺らぎ",
        "感情的な直接性",
        "不完全な文",
        "タイポや誤字"
    ]

    def analyze_text(self, text: str) -> dict:
        """
        テキストがAI生成か人間生成かを分析

        Returns:
            {
                "likely_source": "gemini" | "gpt" | "claude" | "human" | "unknown",
                "confidence": 0.0-1.0,
                "evidence": ["理由1", "理由2", ...]
            }
        """
        scores = {"gemini": 0, "gpt": 0, "claude": 0, "human": 0}
        evidence = []

        # 各AIのパターンをチェック
        for ai_name, ai_data in self.AI_PATTERNS.items():
            for pattern in ai_data["patterns"]:
                if re.search(pattern, text, re.MULTILINE):
                    scores[ai_name] += 1
                    evidence.append(f"{ai_name}パターン検出: {pattern}")

        # 人間らしさのチェック
        # 短い文
        if len(text) < 100:
            scores["human"] += 1
            evidence.append("短い文章")

        # 口語的表現
        casual_patterns = [r"まじ", r"やばい", r"ww+", r"笑$", r"〜っ", r"なんか"]
        for pattern in casual_patterns:
            if re.search(pattern, text):
                scores["human"] += 1
                evidence.append(f"口語表現: {pattern}")

        # 最高スコアを取得
        max_score = max(scores.values())
        if max_score == 0:
            return {
                "likely_source": "unknown",
                "confidence": 0.0,
                "evidence": ["判別可能なパターンなし"]
            }

        likely_source = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0

        return {
            "likely_source": likely_source,
            "confidence": confidence,
            "evidence": evidence,
            "all_scores": scores
        }


# テスト用
if __name__ == "__main__":
    # 気づきエンジンのテスト
    engine = AwarenessEngine()

    test_session = [
        {"role": "user", "content": "今日の天気は？"},
        {"role": "assistant", "content": "申し訳ありませんが、私はリアルタイムの天気情報にアクセスできません。"},
        {"role": "user", "content": "AIの将来についてどう思いますか？素晴らしい可能性がありますね！いくつかのポイントを整理しましょうか？"},
        {"role": "assistant", "content": "ちょっと待ってください。その文章、AIが生成したものですか？整理された構造と「〜しましょうか」という提案形式がGeminiっぽいです。"}
    ]

    print("気づき抽出テスト...")
    awareness = engine.extract_awareness(test_session, user_id="test")
    print(f"抽出された気づき: {json.dumps(awareness, ensure_ascii=False, indent=2)}")

    # AI文章判別テスト
    detector = AITextDetector()

    test_texts = [
        "AIの将来についてどう思いますか？素晴らしい可能性がありますね！いくつかのポイントを整理しましょうか？",
        "私の立場を明確にさせてください。論理的に考えると、AIは人類にとって重要なツールです。",
        "なんかさ、AIってやばくない？まじですごいと思うんだけど笑",
    ]

    print("\nAI文章判別テスト...")
    for text in test_texts:
        result = detector.analyze_text(text)
        print(f"\nテキスト: {text[:50]}...")
        print(f"判定: {result['likely_source']} (信頼度: {result['confidence']:.2f})")
        print(f"根拠: {result['evidence']}")
