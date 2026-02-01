"""
LLM Awareness Monitor - Streamlit UI
チャット + 内面観察 + ユーザー評価を一画面で

Usage:
    streamlit run awareness_ui.py
"""

import streamlit as st
import json
import requests
from pathlib import Path
from datetime import datetime
import time

# パス設定
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
THINKING_HABITS_FILE = DATA_DIR / "thinking_habits" / "integrated_reflections.jsonl"
USER_RATINGS_FILE = DATA_DIR / "user_ratings.jsonl"

# LM Studio設定
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

# ページ設定
st.set_page_config(
    page_title="LLM Awareness Monitor",
    page_icon="🧠",
    layout="wide"
)

# カスタムCSS（ダークモード対応・高コントラスト）
st.markdown("""
<style>
    .chat-user {
        background-color: #1e3a5f;
        color: #ffffff;
        padding: 12px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #64b5f6;
    }
    .chat-bot {
        background-color: #2d1b4e;
        color: #ffffff;
        padding: 12px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #ce93d8;
    }
    .inner-state {
        background-color: #3d2e1f;
        color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffb74d;
    }
    .meta-insight {
        background-color: #4a1942;
        color: #ffffff;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #f06292;
    }
    .gap-positive {
        color: #81c784;
        font-weight: bold;
        font-size: 18px;
    }
    .gap-negative {
        color: #e57373;
        font-weight: bold;
        font-size: 18px;
    }
    .emotion-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


def read_jsonl_entries(file_path: Path, last_n: int = 50) -> list:
    """JSONLファイルから最新のエントリを読み込む"""
    entries = []
    if not file_path.exists():
        return entries

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[-last_n:]:
            try:
                entries.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return entries


def save_user_rating(
    entry_timestamp: str,
    ai_predicted: int,
    user_actual: int,
    gap: int,
    user_comment: str = "",
    full_entry: dict = None
):
    """ユーザー評価を全データと共に保存（夢見モード用）"""
    USER_RATINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    rating_data = {
        "entry_timestamp": entry_timestamp,
        "ai_predicted": ai_predicted,
        "user_actual": user_actual,
        "gap": gap,
        "user_comment": user_comment,
        "rated_at": datetime.now().isoformat(),
        # 夢見モード用：完全なコンテキスト
        "context": {
            "user_input": full_entry.get("user_input", "") if full_entry else "",
            "assistant_output": full_entry.get("assistant_output", "") if full_entry else "",
            "emotion": full_entry.get("emotion", {}) if full_entry else {},
            "background": full_entry.get("background", {}) if full_entry else {},
            "user_perspective": full_entry.get("user_perspective", {}) if full_entry else {},
            "meta_insight": full_entry.get("meta_insight", "") if full_entry else "",
        }
    }

    with open(USER_RATINGS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rating_data, ensure_ascii=False) + "\n")


def get_emotion_color(emotion: str) -> str:
    """感情に応じた色を返す"""
    colors = {
        "empathy": "#9c27b0",
        "共感": "#9c27b0",
        "confident": "#ffc107",
        "自信": "#ffc107",
        "confused": "#ff9800",
        "混乱": "#ff9800",
        "anxious": "#f44336",
        "不安": "#f44336",
        "curious": "#00bcd4",
        "好奇心": "#00bcd4",
        "happy": "#4caf50",
        "楽しい": "#4caf50",
    }
    return colors.get(emotion, "#9e9e9e")


def main():
    st.title("🧠 LLM Awareness Monitor")
    st.caption("チャット + 内面観察 + RLSF評価システム")

    # タブ
    tab1, tab2, tab3 = st.tabs(["📊 リアルタイム監視", "📜 履歴分析", "📈 Gap統計"])

    # ========== Tab 1: リアルタイム監視 ==========
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("💬 会話ログ")

            # 最新エントリを取得
            entries = read_jsonl_entries(THINKING_HABITS_FILE, 10)

            if not entries:
                st.info("まだデータがありません。Botと会話を始めてください。")
            else:
                for i, entry in enumerate(reversed(entries[-5:])):  # 最新5件
                    user_input = entry.get("user_input", "")
                    bot_output = entry.get("assistant_output", "")
                    timestamp = entry.get("timestamp", "")[:19]

                    with st.container():
                        st.markdown(f"**{timestamp}**")
                        st.markdown(f'<div class="chat-user">👤 {user_input}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="chat-bot">🤖 {bot_output}</div>', unsafe_allow_html=True)
                        st.divider()

        with col2:
            st.subheader("🔮 内面状態")

            if entries:
                # 最新のエントリ
                latest = entries[-1]

                emotion = latest.get("emotion", {})
                emotion_label = emotion.get("label", "unknown")
                emotion_note = emotion.get("note", "")
                emotion_color = get_emotion_color(emotion_label)

                background = latest.get("background", {})
                bg_statement = background.get("statement", "")

                user_perspective = latest.get("user_perspective", {})
                ai_predicted = user_perspective.get("satisfaction", 3)
                user_impression = user_perspective.get("impression", "")
                would_improve = user_perspective.get("would_improve", "")

                meta_insight = latest.get("meta_insight", "")
                timestamp = latest.get("timestamp", "")

                # 感情表示
                st.markdown(f"""
                <span class="emotion-badge" style="background-color: {emotion_color}; color: white;">
                    ● {emotion_label}
                </span>
                """, unsafe_allow_html=True)

                if emotion_note:
                    st.caption(f"_{emotion_note}_")

                # 背景
                if bg_statement:
                    st.markdown("**Background:**")
                    st.info(bg_statement)

                # AI予測
                st.markdown("**AI's User Satisfaction Prediction:**")
                st.markdown(f"### {ai_predicted}/5 ⭐")

                if user_impression:
                    st.caption(f"AI thinks: _{user_impression}_")

                # ユーザー評価入力
                st.markdown("---")
                st.markdown("**Your Actual Rating:**")
                user_rating = st.slider(
                    "この応答の実際の満足度は？",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f"rating_{timestamp}"
                )

                # Gap計算
                if isinstance(ai_predicted, (int, float)):
                    gap = user_rating - int(ai_predicted)
                    gap_color = "gap-positive" if gap >= 0 else "gap-negative"
                    gap_text = f"+{gap}" if gap > 0 else str(gap)

                    st.markdown(f"""
                    **Gap:** <span class="{gap_color}">{gap_text}</span>
                    {"(AIが過信)" if gap < 0 else "(AIが過小評価)" if gap > 0 else "(正確)"}
                    """, unsafe_allow_html=True)

                # コメント入力
                user_comment = st.text_area(
                    "コメント（任意）",
                    placeholder="なぜこの評価にしたか、どう改善すべきかなど...",
                    key=f"comment_{timestamp}",
                    height=80
                )

                # 保存ボタン
                if st.button("💾 評価を保存", key=f"save_{timestamp}"):
                    save_user_rating(
                        timestamp,
                        int(ai_predicted),
                        user_rating,
                        gap,
                        user_comment,
                        latest  # 全データを渡す
                    )
                    st.success("評価を保存しました！（夢見モードで内省に使用されます）")

                # 改善提案
                if would_improve:
                    st.markdown("---")
                    st.markdown("**Would Improve:**")
                    st.warning(would_improve)

                # メタ洞察
                if meta_insight:
                    st.markdown("---")
                    st.markdown('<div class="meta-insight">', unsafe_allow_html=True)
                    st.markdown(f"**✨ Meta-Insight:**")
                    st.markdown(f"_{meta_insight}_")
                    st.markdown('</div>', unsafe_allow_html=True)

            # 自動更新
            if st.button("🔄 更新"):
                st.rerun()

    # ========== Tab 2: 履歴分析 ==========
    with tab2:
        st.subheader("📜 会話履歴 + 内面状態")

        entries = read_jsonl_entries(THINKING_HABITS_FILE, 50)

        if not entries:
            st.info("データがありません")
        else:
            # フィルター
            emotions = list(set(e.get("emotion", {}).get("label", "unknown") for e in entries))
            selected_emotion = st.selectbox("感情でフィルター", ["すべて"] + emotions)

            for entry in reversed(entries):
                emotion = entry.get("emotion", {})
                emotion_label = emotion.get("label", "unknown")

                if selected_emotion != "すべて" and emotion_label != selected_emotion:
                    continue

                emotion_color = get_emotion_color(emotion_label)
                timestamp = entry.get("timestamp", "")[:19]
                user_input = entry.get("user_input", "")
                bot_output = entry.get("assistant_output", "")
                ai_predicted = entry.get("user_perspective", {}).get("satisfaction", "?")
                meta_insight = entry.get("meta_insight", "")

                with st.expander(f"[{timestamp}] ● {emotion_label} | 👤 {user_input[:50]}..."):
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown("**User:**")
                        st.write(user_input)
                        st.markdown("**Bot:**")
                        st.write(bot_output)

                    with col2:
                        st.markdown(f"**Emotion:** {emotion_label}")
                        st.markdown(f"**AI Predicted:** {ai_predicted}/5")
                        if meta_insight:
                            st.markdown(f"**Meta-Insight:** _{meta_insight}_")

    # ========== Tab 3: Gap統計 ==========
    with tab3:
        st.subheader("📈 RLSF Gap分析")

        # ユーザー評価データを読み込み
        if USER_RATINGS_FILE.exists():
            ratings = read_jsonl_entries(USER_RATINGS_FILE, 100)

            if ratings:
                import pandas as pd
                df = pd.DataFrame(ratings)

                # 基本統計
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_gap = df["gap"].mean()
                    st.metric("平均Gap", f"{avg_gap:.2f}")
                with col2:
                    overconfident = len(df[df["gap"] < 0])
                    st.metric("AI過信回数", overconfident)
                with col3:
                    accurate = len(df[df["gap"] == 0])
                    st.metric("正確な予測", accurate)

                # Gap推移グラフ
                st.markdown("### Gap推移")
                st.line_chart(df["gap"])

                # AI予測 vs 実際
                st.markdown("### AI予測 vs 実際の評価")
                chart_data = df[["ai_predicted", "user_actual"]]
                st.bar_chart(chart_data)

                # 詳細テーブル
                st.markdown("### 評価詳細")
                st.dataframe(df[["entry_timestamp", "ai_predicted", "user_actual", "gap"]])
            else:
                st.info("まだ評価データがありません。「リアルタイム監視」タブで評価を入力してください。")
        else:
            st.info("評価データファイルがありません")

        # 訓練データエクスポート
        st.markdown("---")
        st.markdown("### 訓練データエクスポート")
        if st.button("📤 RLSF訓練データを生成"):
            if USER_RATINGS_FILE.exists():
                ratings = read_jsonl_entries(USER_RATINGS_FILE, 1000)
                # Gap < -2 のエントリを抽出（AIが大きく過信）
                overconfident = [r for r in ratings if r.get("gap", 0) < -2]
                st.success(f"AI過信エントリ: {len(overconfident)}件を訓練データとして使用可能")
            else:
                st.warning("評価データがありません")


if __name__ == "__main__":
    main()
