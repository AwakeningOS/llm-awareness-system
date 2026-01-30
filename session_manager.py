"""
セッションマネージャー
- 会話ログの管理
- セッション終了検知
- 自己チェックの実行タイミング制御
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, Callable
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# セッション終了を示すキーワード
END_SESSION_KEYWORDS = [
    "終了", "bye", "バイバイ", "また", "じゃあね", "さようなら",
    "おやすみ", "ありがとう、終わり", "これで終わり", "終わりで",
    "quit", "exit", "end", "goodbye", "see you"
]

# セッションタイムアウト（秒）
SESSION_TIMEOUT = 1800  # 30分


class Session:
    """個別セッションの管理"""

    def __init__(self, user_id: str, user_name: str = "Unknown"):
        self.user_id = user_id
        self.user_name = user_name
        self.messages: list[dict] = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.is_active = True
        self.metadata: dict = {}

    def add_message(self, role: str, content: str):
        """メッセージを追加"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity = datetime.now()

    def get_messages_for_extraction(self) -> list[dict]:
        """気づき抽出用のメッセージリストを取得"""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def is_expired(self, timeout_seconds: int = SESSION_TIMEOUT) -> bool:
        """セッションが期限切れかチェック"""
        return datetime.now() - self.last_activity > timedelta(seconds=timeout_seconds)

    def should_end(self, last_message: str) -> bool:
        """セッション終了条件をチェック"""
        last_message_lower = last_message.lower().strip()
        for keyword in END_SESSION_KEYWORDS:
            if keyword in last_message_lower:
                return True
        return False

    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active,
            "metadata": self.metadata
        }


class SessionManager:
    """セッション全体の管理"""

    def __init__(
        self,
        timeout_seconds: int = SESSION_TIMEOUT,
        on_session_end: Optional[Callable] = None,
        session_log_dir: Optional[Path] = None
    ):
        """
        Args:
            timeout_seconds: セッションタイムアウト（秒）
            on_session_end: セッション終了時のコールバック
            session_log_dir: セッションログの保存ディレクトリ
        """
        self.sessions: dict[str, Session] = {}
        self.timeout_seconds = timeout_seconds
        self.on_session_end = on_session_end
        self.session_log_dir = session_log_dir
        self._cleanup_task: Optional[asyncio.Task] = None
        self._pending_end_sessions: set[str] = set()  # イベントループなしで終了リクエストされたセッション

        if session_log_dir:
            session_log_dir.mkdir(parents=True, exist_ok=True)

    def get_or_create_session(self, user_id: str, user_name: str = "Unknown") -> Session:
        """セッションを取得または作成"""
        if user_id not in self.sessions or not self.sessions[user_id].is_active:
            self.sessions[user_id] = Session(user_id, user_name)
            logger.info(f"新規セッション作成: {user_id}")
        return self.sessions[user_id]

    def add_message(self, user_id: str, role: str, content: str, user_name: str = "Unknown"):
        """メッセージを追加"""
        session = self.get_or_create_session(user_id, user_name)
        session.add_message(role, content)

        # セッション終了チェック
        if role == "user" and session.should_end(content):
            logger.info(f"セッション終了キーワード検出: {user_id}")
            # イベントループが存在する場合のみタスクを作成
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._end_session(user_id))
            except RuntimeError:
                # イベントループがない場合は、フラグを立てて後で処理
                self._pending_end_sessions.add(user_id)
                logger.warning(f"イベントループなし、セッション終了をペンディング: {user_id}")

    async def _end_session(self, user_id: str):
        """セッションを終了"""
        if user_id not in self.sessions:
            return

        session = self.sessions[user_id]
        session.is_active = False

        # セッションログを保存
        if self.session_log_dir:
            self._save_session_log(session)

        # コールバック実行
        if self.on_session_end:
            try:
                if asyncio.iscoroutinefunction(self.on_session_end):
                    await self.on_session_end(session)
                else:
                    self.on_session_end(session)
            except Exception as e:
                logger.error(f"セッション終了コールバックエラー: {e}")

        logger.info(f"セッション終了: {user_id} (メッセージ数: {len(session.messages)})")

    def _save_session_log(self, session: Session):
        """セッションログを保存"""
        if not self.session_log_dir:
            return

        filename = f"{session.user_id}_{session.created_at.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.session_log_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"セッションログ保存: {filepath}")

    async def start_cleanup_task(self):
        """期限切れセッションのクリーンアップタスクを開始"""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """定期的に期限切れセッションをチェック"""
        while True:
            await asyncio.sleep(60)  # 1分ごとにチェック
            await self._cleanup_expired_sessions()
            await self._process_pending_end_sessions()

    async def _process_pending_end_sessions(self):
        """ペンディング中のセッション終了を処理"""
        if not self._pending_end_sessions:
            return

        pending = self._pending_end_sessions.copy()
        self._pending_end_sessions.clear()

        for user_id in pending:
            logger.info(f"ペンディングセッション終了を処理: {user_id}")
            await self._end_session(user_id)

    async def _cleanup_expired_sessions(self):
        """期限切れセッションを終了"""
        expired_users = []
        for user_id, session in self.sessions.items():
            if session.is_active and session.is_expired(self.timeout_seconds):
                expired_users.append(user_id)

        for user_id in expired_users:
            logger.info(f"セッションタイムアウト: {user_id}")
            await self._end_session(user_id)

    def get_session(self, user_id: str) -> Optional[Session]:
        """セッションを取得"""
        return self.sessions.get(user_id)

    def get_active_session_count(self) -> int:
        """アクティブなセッション数を取得"""
        return sum(1 for s in self.sessions.values() if s.is_active)

    def clear_session(self, user_id: str):
        """セッションをクリア（会話履歴リセット）"""
        if user_id in self.sessions:
            self.sessions[user_id].messages = []
            self.sessions[user_id].last_activity = datetime.now()
            logger.info(f"セッションクリア: {user_id}")

    async def force_end_session(self, user_id: str):
        """セッションを強制終了（気づき抽出をトリガー）"""
        await self._end_session(user_id)

    def stop_cleanup_task(self):
        """クリーンアップタスクを停止"""
        if self._cleanup_task:
            self._cleanup_task.cancel()


# テスト用
if __name__ == "__main__":
    import asyncio

    async def on_session_end(session: Session):
        print(f"セッション終了コールバック: {session.user_id}")
        print(f"メッセージ数: {len(session.messages)}")
        for msg in session.messages:
            print(f"  {msg['role']}: {msg['content'][:50]}...")

    async def test():
        manager = SessionManager(
            timeout_seconds=5,  # テスト用に短く
            on_session_end=on_session_end
        )

        # メッセージ追加テスト
        manager.add_message("user1", "user", "こんにちは", "テストユーザー")
        manager.add_message("user1", "assistant", "こんにちは！何かお手伝いできますか？")
        manager.add_message("user1", "user", "AIについて教えて")
        manager.add_message("user1", "assistant", "AIは人工知能の略で...")

        print(f"アクティブセッション数: {manager.get_active_session_count()}")

        # 終了キーワードテスト
        manager.add_message("user1", "user", "ありがとう、また！")

        # 少し待つ
        await asyncio.sleep(1)

        print(f"アクティブセッション数: {manager.get_active_session_count()}")

    asyncio.run(test())
