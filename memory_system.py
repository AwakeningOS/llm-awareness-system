"""
ChromaDB ベクトル記憶システム
- Memory MCP（知識グラフ）と併用する
- 曖昧な検索・類似度検索に強い
"""

import chromadb
from chromadb.config import Settings
from datetime import datetime
from pathlib import Path
from typing import Optional


class MemorySystem:
    """ベクトルベースの長期記憶システム"""

    def __init__(self, data_dir: str = "./data/chromadb"):
        """
        Args:
            data_dir: ChromaDBのデータ保存ディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.data_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"description": "会話履歴とユーザー情報のベクトル記憶"}
        )

    def save(
        self,
        content: str,
        category: str = "general",
        importance: int = 5,
        user_id: str = "default",
        metadata: Optional[dict] = None
    ) -> str:
        """
        記憶を保存する

        Args:
            content: 保存する内容
            category: カテゴリ (user_info, preference, event, emotion, conversation)
            importance: 重要度 (1-10)
            user_id: ユーザーID
            metadata: 追加のメタデータ

        Returns:
            保存した記憶のID
        """
        memory_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        doc_metadata = {
            "category": category,
            "importance": importance,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
        }

        if metadata:
            doc_metadata.update(metadata)

        self.collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[doc_metadata]
        )

        return memory_id

    def search(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 5,
        category: Optional[str] = None
    ) -> list[dict]:
        """
        類似度検索で記憶を検索する

        Args:
            query: 検索クエリ
            user_id: ユーザーID
            limit: 取得件数
            category: フィルタするカテゴリ（Noneなら全カテゴリ）

        Returns:
            検索結果のリスト
        """
        where_filter = {"user_id": user_id}
        if category:
            where_filter["category"] = category

        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_filter
        )

        memories = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                memories.append({
                    "id": results["ids"][0][i],
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None
                })

        return memories

    def get_recent(
        self,
        user_id: str = "default",
        limit: int = 10
    ) -> list[dict]:
        """
        最近の記憶を取得する

        Args:
            user_id: ユーザーID
            limit: 取得件数

        Returns:
            最近の記憶のリスト
        """
        results = self.collection.get(
            where={"user_id": user_id},
            limit=limit
        )

        memories = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                memories.append({
                    "id": results["ids"][i],
                    "content": doc,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                })

        # created_atでソート（新しい順）
        memories.sort(
            key=lambda x: x["metadata"].get("created_at", ""),
            reverse=True
        )

        return memories[:limit]

    def delete(self, memory_id: str) -> bool:
        """
        記憶を削除する

        Args:
            memory_id: 削除する記憶のID

        Returns:
            成功したかどうか
        """
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    def count(self, user_id: Optional[str] = None) -> int:
        """
        記憶の総数を取得する

        Args:
            user_id: ユーザーID（Noneなら全体）

        Returns:
            記憶の数
        """
        if user_id:
            results = self.collection.get(where={"user_id": user_id})
            return len(results["ids"])
        return self.collection.count()


# テスト用
if __name__ == "__main__":
    memory = MemorySystem()

    # テスト保存
    memory_id = memory.save(
        content="ユーザーはケーキが好きです",
        category="preference",
        importance=8,
        user_id="test_user"
    )
    print(f"保存完了: {memory_id}")

    # テスト検索
    results = memory.search("好きな食べ物", user_id="test_user")
    print(f"検索結果: {results}")

    # 件数確認
    count = memory.count(user_id="test_user")
    print(f"記憶数: {count}")
