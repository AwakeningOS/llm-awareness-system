"""
気づきデータベース
- JSONL形式で蓄積
- メタデータ管理
- 重複チェック
- 学習データのエクスポート
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import hashlib
import logging

logger = logging.getLogger(__name__)


class AwarenessDatabase:
    """気づきデータベース"""

    def __init__(self, data_dir: str = "./data/awareness"):
        """
        Args:
            data_dir: データ保存ディレクトリ
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # メインの気づきファイル
        self.awareness_file = self.data_dir / "awareness.jsonl"

        # 学習データファイル
        self.training_file = self.data_dir / "training_data.jsonl"

        # 統計ファイル
        self.stats_file = self.data_dir / "stats.json"

        # 重複チェック用のハッシュセット
        self._content_hashes: set[str] = set()
        self._load_hashes()

    def _load_hashes(self):
        """既存データのハッシュを読み込む"""
        if self.awareness_file.exists():
            with open(self.awareness_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        content_hash = self._compute_hash(data)
                        self._content_hashes.add(content_hash)
                    except json.JSONDecodeError:
                        continue

    def _compute_hash(self, data: dict) -> str:
        """データのハッシュを計算"""
        # description + my_response で重複判定
        content = f"{data.get('description', '')}{data.get('my_response', '')}"
        return hashlib.md5(content.encode()).hexdigest()

    def save_awareness(self, awareness: dict) -> bool:
        """
        気づきを保存

        Args:
            awareness: 気づきデータ

        Returns:
            保存成功したか（重複の場合はFalse）
        """
        # 重複チェック
        content_hash = self._compute_hash(awareness)
        if content_hash in self._content_hashes:
            logger.info("重複する気づきをスキップ")
            return False

        # 保存
        with open(self.awareness_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(awareness, ensure_ascii=False) + "\n")

        self._content_hashes.add(content_hash)
        logger.info(f"気づきを保存: {awareness.get('type', 'unknown')}")

        # 統計を更新
        self._update_stats(awareness)

        return True

    def save_training_data(self, training_data: dict) -> bool:
        """
        学習データを保存

        Args:
            training_data: 学習データ（messages + metadata形式）

        Returns:
            保存成功したか
        """
        with open(self.training_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(training_data, ensure_ascii=False) + "\n")

        logger.info("学習データを保存")
        return True

    def _update_stats(self, awareness: dict):
        """統計を更新"""
        stats = self.get_stats()

        # カウント更新
        stats["total_count"] = stats.get("total_count", 0) + 1
        stats["last_updated"] = datetime.now().isoformat()

        # タイプ別カウント
        awareness_type = awareness.get("type", "unknown")
        if "by_type" not in stats:
            stats["by_type"] = {}
        stats["by_type"][awareness_type] = stats["by_type"].get(awareness_type, 0) + 1

        # カテゴリ別カウント
        category = awareness.get("category", "unknown")
        if "by_category" not in stats:
            stats["by_category"] = {}
        stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

        # スコア分布
        score = awareness.get("learning_potential", 3)
        if "by_score" not in stats:
            stats["by_score"] = {}
        stats["by_score"][str(score)] = stats["by_score"].get(str(score), 0) + 1

        # 保存
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> dict:
        """統計を取得"""
        if self.stats_file.exists():
            with open(self.stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "total_count": 0,
            "by_type": {},
            "by_category": {},
            "by_score": {},
            "last_updated": None
        }

    def count(self) -> int:
        """気づきの総数を取得"""
        stats = self.get_stats()
        return stats.get("total_count", 0)

    def count_training_data(self) -> int:
        """学習データの数を取得"""
        if not self.training_file.exists():
            return 0
        with open(self.training_file, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def get_all_awareness(self, limit: int = 100) -> list[dict]:
        """全ての気づきを取得"""
        awareness_list = []
        if not self.awareness_file.exists():
            return awareness_list

        with open(self.awareness_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    awareness_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        # 新しい順にソート
        awareness_list.sort(
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )

        return awareness_list[:limit]

    def get_by_type(self, awareness_type: str, limit: int = 50) -> list[dict]:
        """タイプ別に気づきを取得"""
        all_awareness = self.get_all_awareness(limit=1000)
        filtered = [a for a in all_awareness if a.get("type") == awareness_type]
        return filtered[:limit]

    def get_by_category(self, category: str, limit: int = 50) -> list[dict]:
        """カテゴリ別に気づきを取得"""
        all_awareness = self.get_all_awareness(limit=1000)
        filtered = [a for a in all_awareness if a.get("category") == category]
        return filtered[:limit]

    def get_high_quality(self, min_score: int = 4, limit: int = 50) -> list[dict]:
        """高品質な気づきを取得"""
        all_awareness = self.get_all_awareness(limit=1000)
        filtered = [a for a in all_awareness if a.get("learning_potential", 0) >= min_score]
        return filtered[:limit]

    def export_training_data(
        self,
        output_path: Optional[Path] = None,
        min_score: int = 3
    ) -> Path:
        """
        学習データをエクスポート

        Args:
            output_path: 出力パス（Noneならデフォルト）
            min_score: 最低スコア

        Returns:
            エクスポートしたファイルパス
        """
        if output_path is None:
            output_path = self.data_dir / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        if not self.training_file.exists():
            logger.warning("学習データファイルが存在しません")
            return output_path

        with open(self.training_file, "r", encoding="utf-8") as f_in:
            with open(output_path, "w", encoding="utf-8") as f_out:
                for line in f_in:
                    try:
                        data = json.loads(line)
                        if data.get("metadata", {}).get("score", 0) >= min_score:
                            f_out.write(line)
                    except json.JSONDecodeError:
                        continue

        logger.info(f"学習データをエクスポート: {output_path}")
        return output_path

    def is_ready_for_training(self, min_samples: int = 100) -> bool:
        """LoRA学習の準備ができているか"""
        return self.count_training_data() >= min_samples

    def get_training_readiness(self, min_samples: int = 100) -> dict:
        """学習準備状況を取得"""
        current = self.count_training_data()
        return {
            "current_samples": current,
            "required_samples": min_samples,
            "ready": current >= min_samples,
            "progress_percent": min(100, (current / min_samples) * 100)
        }


# テスト用
if __name__ == "__main__":
    db = AwarenessDatabase(data_dir="./data/awareness_test")

    # テストデータ保存
    test_awareness = {
        "awareness_detected": True,
        "type": "自発的気づき",
        "category": "AI文章判別",
        "description": "ユーザーの入力がAI生成文章であることに気づいた",
        "trigger": "整理された構造と提案形式の文章",
        "my_response": "その文章、AIが生成したものですか？",
        "significance": "入力の生成主体を認識する能力",
        "learning_potential": 5,
        "timestamp": datetime.now().isoformat(),
        "user_id": "test_user"
    }

    result = db.save_awareness(test_awareness)
    print(f"保存結果: {result}")

    # 統計表示
    stats = db.get_stats()
    print(f"統計: {json.dumps(stats, ensure_ascii=False, indent=2)}")

    # 学習準備状況
    readiness = db.get_training_readiness(min_samples=100)
    print(f"学習準備状況: {readiness}")
