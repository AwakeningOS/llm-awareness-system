"""
ChromaDB Vector Memory System
- Used alongside Memory MCP (knowledge graph)
- Strong for fuzzy search and similarity search
"""

import chromadb
from chromadb.config import Settings
from datetime import datetime
from pathlib import Path
from typing import Optional


class MemorySystem:
    """Vector-based long-term memory system"""

    def __init__(self, data_dir: str = "./data/chromadb"):
        """
        Args:
            data_dir: ChromaDB data storage directory
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.data_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"description": "Vector memory for conversation history and user information"}
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
        Save memory

        Args:
            content: Content to save
            category: Category (user_info, preference, event, emotion, conversation)
            importance: Importance level (1-10)
            user_id: User ID
            metadata: Additional metadata

        Returns:
            ID of saved memory
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
        Search memory by similarity

        Args:
            query: Search query
            user_id: User ID
            limit: Number of results to retrieve
            category: Category to filter (None for all categories)

        Returns:
            List of search results
        """
        # Build where filter (ChromaDB requires $and for multiple conditions)
        if category:
            where_filter = {
                "$and": [
                    {"user_id": user_id},
                    {"category": category}
                ]
            }
        else:
            where_filter = {"user_id": user_id}

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
        Get recent memories

        Args:
            user_id: User ID
            limit: Number of results to retrieve

        Returns:
            List of recent memories
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

        # Sort by created_at (newest first)
        memories.sort(
            key=lambda x: x["metadata"].get("created_at", ""),
            reverse=True
        )

        return memories[:limit]

    def delete(self, memory_id: str) -> bool:
        """
        Delete memory

        Args:
            memory_id: ID of memory to delete

        Returns:
            Whether deletion was successful
        """
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    def count(self, user_id: Optional[str] = None) -> int:
        """
        Get total memory count

        Args:
            user_id: User ID (None for all)

        Returns:
            Memory count
        """
        if user_id:
            results = self.collection.get(where={"user_id": user_id})
            return len(results["ids"])
        return self.collection.count()


# Test code
if __name__ == "__main__":
    memory = MemorySystem()

    # Test save
    memory_id = memory.save(
        content="User likes cake",
        category="preference",
        importance=8,
        user_id="test_user"
    )
    print(f"Save complete: {memory_id}")

    # Test search
    results = memory.search("favorite food", user_id="test_user")
    print(f"Search results: {results}")

    # Count check
    count = memory.count(user_id="test_user")
    print(f"Memory count: {count}")
