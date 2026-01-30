"""
VOICEVOX 音声合成モジュール
"""

import requests
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class VoicevoxClient:
    """VOICEVOX音声合成クライアント"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50021,
        speaker_id: int = 1
    ):
        """
        Args:
            host: VOICEVOXサーバーのホスト
            port: VOICEVOXサーバーのポート
            speaker_id: 話者ID（デフォルト: 1）
        """
        self.base_url = f"http://{host}:{port}"
        self.speaker_id = speaker_id

    def is_available(self) -> bool:
        """VOICEVOXサーバーが利用可能か確認"""
        try:
            response = requests.get(f"{self.base_url}/speakers", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def get_speakers(self) -> list[dict]:
        """利用可能な話者一覧を取得"""
        try:
            response = requests.get(f"{self.base_url}/speakers", timeout=5)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None
    ) -> Optional[bytes]:
        """
        テキストを音声に変換する

        Args:
            text: 読み上げるテキスト
            speaker_id: 話者ID（Noneならデフォルトを使用）

        Returns:
            WAV形式の音声データ（失敗時はNone）
        """
        speaker = speaker_id or self.speaker_id

        try:
            # 音声合成用のクエリを作成
            query_response = requests.post(
                f"{self.base_url}/audio_query",
                params={"text": text, "speaker": speaker},
                timeout=30
            )

            if query_response.status_code != 200:
                return None

            query_data = query_response.json()

            # 音声を合成
            synthesis_response = requests.post(
                f"{self.base_url}/synthesis",
                params={"speaker": speaker},
                json=query_data,
                timeout=60
            )

            if synthesis_response.status_code != 200:
                return None

            return synthesis_response.content

        except Exception as e:
            print(f"VOICEVOX合成エラー: {e}")
            return None

    def speak(
        self,
        text: str,
        speaker_id: Optional[int] = None
    ) -> bool:
        """
        テキストを音声合成して再生する

        Args:
            text: 読み上げるテキスト
            speaker_id: 話者ID

        Returns:
            成功したかどうか
        """
        audio_data = self.synthesize(text, speaker_id)

        if not audio_data:
            return False

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            # PowerShellで再生（Windows）
            subprocess.run(
                [
                    "powershell",
                    "-Command",
                    f'(New-Object Media.SoundPlayer "{temp_path}").PlaySync()'
                ],
                check=True,
                timeout=60
            )
            return True
        except Exception as e:
            print(f"音声再生エラー: {e}")
            return False
        finally:
            # 一時ファイルを削除
            try:
                Path(temp_path).unlink()
            except Exception:
                pass


# テスト用
if __name__ == "__main__":
    client = VoicevoxClient()

    if client.is_available():
        print("VOICEVOXサーバーに接続しました")
        speakers = client.get_speakers()
        print(f"利用可能な話者数: {len(speakers)}")

        # テスト発話
        client.speak("こんにちは、テストです")
    else:
        print("VOICEVOXサーバーに接続できません")
