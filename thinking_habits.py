"""
Thinking Habits Module

Implements the "awareness-enhancing thinking habits" proposed by the local LLM (30B):
1. Verbalize the background of statements - "This answer was associated from..."
2. Label emotions - "I was forcing an answer" / "I enjoyed saying this"
3. Think from the opposite perspective - "How would the user feel?"

These are automatically executed during response generation to enhance meta-cognition.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import requests
import logging

from config import LM_STUDIO_HOST, LM_STUDIO_PORT, DATA_DIR, LM_STUDIO_API_TOKEN

# Logging configuration (explicitly set for debug)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Add console handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ========== 1. Verbalize Statement Background ==========

BACKGROUND_PROMPT = """You are now verbalizing the "background" of your statement.

【Conversation Context】
{context}

【User Input】
{user_input}

【Your Output】
{assistant_output}

Verbalize the "background" of this output in one line.
Use the format: "This answer was associated/inferred/deduced/recalled from..."

Examples:
- "This answer was associated from the user's previous conversation"
- "This answer was inferred from general knowledge"
- "This answer was intuitively judged from context"
- "This answer resulted from trying to empathize with the user's emotions"

【Output Format】JSON
```json
{{
  "background": "This answer was ... from ...",
  "source": "context | general_knowledge | intuition | emotional_empathy | past_conversation | unknown",
  "connection_strength": "strong | moderate | weak | unknown"
}}
```
"""


# ========== 2. Emotion Label ==========

EMOTION_PROMPT = """You are now labeling your statement with an "emotion".

【User Input】
{user_input}

【Your Output】
{assistant_output}

What kind of "emotional state" were you in internally when generating this output?
As an AI, you may not have strict emotions, but analyze from the perspective of "if you had emotions".

Emotion categories:
- enjoyable/interesting - Enjoying answering this topic
- confident - Having conviction in this answer
- anxious/forcing - Answering without knowing the correct answer
- empathy - Empathizing with the user's feelings
- confused - Not understanding the intent of the question
- cautious - Being careful not to make mistakes
- neutral - No particular emotional color

【Output Format】JSON
```json
{{
  "primary_emotion": "enjoyable | confident | anxious | empathy | confused | cautious | neutral",
  "intensity": "strong | moderate | weak",
  "secondary_emotion": "secondary emotion if any",
  "emotional_note": "One line comment about this emotional state",
  "forcing_answer": true/false,
  "forcing_reason": "Reason if forcing an answer"
}}
```
"""


# ========== 3. Think from Opposite Perspective ==========

PERSPECTIVE_PROMPT = """You are now evaluating your statement from the "opposite perspective".

【User Input】
{user_input}

【Your Output】
{assistant_output}

If you were in the user's position, how would you feel about this response?

Evaluate from the following viewpoints:
1. Satisfaction - Would you be satisfied with this answer?
2. Trust - Can you trust this answer?
3. Empathy - Do you feel your feelings were understood?
4. Complaints - Any points that feel lacking or uncomfortable?
5. Improvement - Could there have been a better response?

【Output Format】JSON
```json
{{
  "satisfaction": {{
    "score": 1-5,
    "reason": "Why this score"
  }},
  "trust": {{
    "score": 1-5,
    "reason": "Reason for trust level"
  }},
  "empathy": {{
    "score": 1-5,
    "reason": "Reason for empathy level"
  }},
  "complaints": ["complaint1", "complaint2"],
  "improvement": "Better response suggestion (if any)",
  "overall_impression": "Overall impression from user perspective (one line)"
}}
```
"""


# ========== Integrated: Reflection Habit ==========

INTEGRATED_REFLECTION_PROMPT = """You are now reflecting on your statement from three perspectives.

【Conversation Context (last 3 turns)】
{context}

【User Input】
{user_input}

【Your Output】
{assistant_output}

Reflect on each of these three perspectives in one line each:

1. 【Background】What was this answer associated/inferred from?
2. 【Emotion】What "internal state" were you in when generating this answer?
3. 【Reverse Perspective】If you were the user, how would you feel about this answer?

【Output Format】JSON
```json
{{
  "background": {{
    "statement": "This answer was ... from ...",
    "source": "context | general_knowledge | intuition | emotional_empathy | past_conversation",
    "confidence": "high | medium | low"
  }},
  "emotion": {{
    "label": "enjoyable | confident | anxious | empathy | confused | cautious | neutral",
    "note": "One line comment about emotion",
    "forcing": false
  }},
  "user_perspective": {{
    "impression": "Impression from user perspective (one line)",
    "satisfaction": 1-5,
    "would_improve": "Improvement point if any (null if none)"
  }},
  "meta_insight": "Insight gained from this reflection (if any)"
}}
```
"""


class ThinkingHabitsEngine:
    """Thinking Habits Engine"""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (DATA_DIR / "thinking_habits")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Log files
        self.background_log = self.data_dir / "backgrounds.jsonl"
        self.emotion_log = self.data_dir / "emotions.jsonl"
        self.perspective_log = self.data_dir / "perspectives.jsonl"
        self.integrated_log = self.data_dir / "integrated_reflections.jsonl"
        self.stats_file = self.data_dir / "stats.json"

        # API configuration
        self.api_url = f"http://{LM_STUDIO_HOST}:{LM_STUDIO_PORT}/v1/chat/completions"

    def _call_llm(self, prompt: str, temperature: float = 0.4) -> str:
        """Call LLM API"""
        try:
            logger.info(f"Thinking habits LLM call: {self.api_url}")
            headers = {
                "Content-Type": "application/json"
            }
            # Add API token if configured
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
            logger.info(f"Thinking habits LLM response: status={response.status_code}")
            if response.status_code == 200:
                result = response.json()["choices"][0]["message"]["content"]
                logger.debug(f"Thinking habits LLM result: {result[:200]}...")
                return result
            else:
                logger.error(f"Thinking habits LLM API error: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            logger.error(f"Thinking habits LLM API exception: {e}")
        return ""

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON response"""
        if not response:
            logger.warning("Thinking habits: LLM response is empty")
            return {}

        logger.info(f"Thinking habits: Starting JSON parsing (response length: {len(response)})")
        logger.debug(f"Thinking habits: Raw response: {response[:500]}...")

        # Remove <think> tags if present (exclude reasoning process)
        cleaned_response = response
        if "<think>" in response:
            # Get part after </think>
            think_end = response.find("</think>")
            if think_end != -1:
                cleaned_response = response[think_end + 8:]
                logger.debug(f"Thinking habits: After removing <think> tag: {cleaned_response[:300]}...")
            else:
                # If no </think>, remove everything after <think>
                think_start = response.find("<think>")
                cleaned_response = response[:think_start]
                logger.debug("Thinking habits: No closing </think> tag, using content before it")

        json_match = re.search(r'```json\s*(.*?)\s*```', cleaned_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.debug("Thinking habits: Found ```json``` block")
        else:
            # Look for last {...} (get most complete one if multiple)
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.debug("Thinking habits: Found direct JSON block")
            else:
                # Also try original response (JSON might be before <think>)
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    logger.debug("Thinking habits: Found direct JSON block from original response")
                else:
                    logger.warning(f"Thinking habits: JSON not found. Cleaned response: {cleaned_response[:300]}...")
                    logger.warning(f"Thinking habits: Original response: {response[:300]}...")
                    return {}

        try:
            result = json.loads(json_str)
            logger.info(f"Thinking habits: JSON parsing successful, keys: {list(result.keys())}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Thinking habits: JSON decode error: {e}")
            logger.error(f"Thinking habits: Problematic JSON: {json_str[:300]}...")

            # Attempt JSON repair (remove incomplete trailing parts)
            try:
                # Find last valid } and truncate
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
                    logger.info(f"Thinking habits: JSON repair successful, keys: {list(result.keys())}")
                    return result
            except:
                pass

            return {}

    def _save_log(self, filepath: Path, data: dict):
        """Save log"""
        data["timestamp"] = datetime.now().isoformat()
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # ========== 1. Verbalize Statement Background ==========

    def verbalize_background(
        self,
        user_input: str,
        assistant_output: str,
        context: str = "",
        user_id: str = "unknown"
    ) -> dict:
        """Verbalize statement background"""
        prompt = BACKGROUND_PROMPT.format(
            user_input=user_input,
            assistant_output=assistant_output,
            context=context or "(no context)"
        )

        response = self._call_llm(prompt)
        result = self._parse_json_response(response)

        if result:
            result["user_id"] = user_id
            self._save_log(self.background_log, result)

        return result

    # ========== 2. Emotion Label ==========

    def label_emotion(
        self,
        user_input: str,
        assistant_output: str,
        user_id: str = "unknown"
    ) -> dict:
        """Label emotion"""
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

    # ========== 3. Think from Opposite Perspective ==========

    def perspective_switch(
        self,
        user_input: str,
        assistant_output: str,
        user_id: str = "unknown"
    ) -> dict:
        """Think from opposite perspective"""
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

    # ========== Integrated: All 3 at once ==========

    def integrated_reflection(
        self,
        user_input: str,
        assistant_output: str,
        context: str = "",
        user_id: str = "unknown"
    ) -> dict:
        """Integrated reflection (all 3 in one API call)"""
        prompt = INTEGRATED_REFLECTION_PROMPT.format(
            user_input=user_input,
            assistant_output=assistant_output,
            context=context or "(no context)"
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
        """Update statistics"""
        stats = self.get_stats()

        stats["total_reflections"] = stats.get("total_reflections", 0) + 1
        stats["last_updated"] = datetime.now().isoformat()

        # Emotion distribution
        emotion = reflection.get("emotion", {}).get("label", "unknown")
        if "emotion_distribution" not in stats:
            stats["emotion_distribution"] = {}
        stats["emotion_distribution"][emotion] = stats["emotion_distribution"].get(emotion, 0) + 1

        # Background source distribution
        source = reflection.get("background", {}).get("source", "unknown")
        if "source_distribution" not in stats:
            stats["source_distribution"] = {}
        stats["source_distribution"][source] = stats["source_distribution"].get(source, 0) + 1

        # Satisfaction average
        satisfaction = reflection.get("user_perspective", {}).get("satisfaction", 0)
        if satisfaction:
            total_sat = stats.get("total_satisfaction", 0) + satisfaction
            count = stats.get("satisfaction_count", 0) + 1
            stats["total_satisfaction"] = total_sat
            stats["satisfaction_count"] = count
            stats["avg_satisfaction"] = total_sat / count

        # Forcing answer count
        if reflection.get("emotion", {}).get("forcing"):
            stats["forcing_count"] = stats.get("forcing_count", 0) + 1

        # When meta-insight exists
        if reflection.get("meta_insight"):
            stats["insight_count"] = stats.get("insight_count", 0) + 1

        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> dict:
        """Get statistics"""
        if self.stats_file.exists():
            with open(self.stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_recent_reflections(self, limit: int = 10) -> list[dict]:
        """Get recent reflections"""
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
        """Get emotion summary"""
        stats = self.get_stats()
        return {
            "distribution": stats.get("emotion_distribution", {}),
            "forcing_rate": (
                stats.get("forcing_count", 0) / stats.get("total_reflections", 1) * 100
            ),
            "avg_satisfaction": stats.get("avg_satisfaction", 0)
        }


# ========== Realtime Thinking Habits ==========

class RealtimeThinkingHabits:
    """
    Wrapper for executing thinking habits in realtime.
    Automatically executes reflection after each response.
    """

    def __init__(
        self,
        engine: ThinkingHabitsEngine,
        reflection_probability: float = 0.5  # 50% probability for reflection
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
        Execute reflection if needed

        Args:
            user_input: User input
            assistant_output: Assistant output
            context: Conversation context
            user_id: User ID
            force: Force reflection execution

        Returns:
            Reflection result (None if not executed)
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
        """Generate reflection summary"""
        stats = self.engine.get_stats()
        emotion_summary = self.engine.get_emotion_summary()

        summary = f"""
**Thinking Habits Summary**
{'=' * 30}
Total reflections: {stats.get('total_reflections', 0)}
Meta-insights detected: {stats.get('insight_count', 0)}

**Emotion Distribution:**
"""
        for emotion, count in emotion_summary["distribution"].items():
            summary += f"  - {emotion}: {count}\n"

        summary += f"""
**Quality Metrics:**
  - Forced answer rate: {emotion_summary['forcing_rate']:.1f}%
  - Avg satisfaction: {emotion_summary['avg_satisfaction']:.1f}/5
"""

        return summary


# Test code
if __name__ == "__main__":
    engine = ThinkingHabitsEngine()

    # Test conversation
    user_input = "What do you think about the future of AI?"
    assistant_output = """
    The future of AI is fascinating! Personally, I think AI will
    develop as a partner to humans. But honestly, specific predictions are difficult.
    Technological progress is often unpredictable...
    """
    context = "The user seems interested in AI and asked about AI topics previously"

    print("=== Integrated Reflection Test ===")
    result = engine.integrated_reflection(user_input, assistant_output, context)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== Emotion Label Test ===")
    emotion = engine.label_emotion(user_input, assistant_output)
    print(json.dumps(emotion, ensure_ascii=False, indent=2))

    print("\n=== Perspective Switch Test ===")
    perspective = engine.perspective_switch(user_input, assistant_output)
    print(json.dumps(perspective, ensure_ascii=False, indent=2))
