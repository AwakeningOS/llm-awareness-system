"""
LoRA学習トリガー
- 一定量の蓄積で通知
- 人間の承認で学習実行
- 学習スクリプトの生成
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

from awareness_database import AwarenessDatabase

logger = logging.getLogger(__name__)

# デフォルトのLoRA設定
DEFAULT_LORA_CONFIG = {
    "base_model": "google/gemma-3-4b-pt",
    "r": 32,
    "lora_alpha": 64,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "epochs": 3,
    "learning_rate": 2e-4,
    "min_samples": 100,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "save_steps": 100,
    "logging_steps": 10,
}


class LoRATrainer:
    """LoRA学習管理"""

    def __init__(
        self,
        database: AwarenessDatabase,
        config: Optional[dict] = None,
        output_dir: str = "./data/lora_adapters"
    ):
        """
        Args:
            database: 気づきデータベース
            config: LoRA設定（Noneならデフォルト）
            output_dir: アダプター出力ディレクトリ
        """
        self.db = database
        self.config = config or DEFAULT_LORA_CONFIG
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 学習スクリプト保存ディレクトリ
        self.scripts_dir = self.output_dir / "scripts"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

    def check_readiness(self) -> dict:
        """学習準備状況をチェック"""
        return self.db.get_training_readiness(self.config["min_samples"])

    def prepare_training_data(self, min_score: int = 3) -> Path:
        """学習データを準備"""
        export_path = self.db.export_training_data(min_score=min_score)
        return export_path

    def generate_training_script(
        self,
        training_data_path: Path,
        output_name: Optional[str] = None
    ) -> Path:
        """
        学習スクリプトを生成

        Args:
            training_data_path: 学習データのパス
            output_name: 出力アダプター名

        Returns:
            生成したスクリプトのパス
        """
        if output_name is None:
            output_name = f"awareness_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        adapter_output_dir = self.output_dir / output_name

        script_content = f'''#!/bin/bash
# 気づき創発システム - LoRA学習スクリプト
# 生成日時: {datetime.now().isoformat()}
# 学習データ: {training_data_path}

# 依存関係のインストール（必要に応じて）
# pip install transformers peft accelerate bitsandbytes datasets

python -m torch.distributed.launch --nproc_per_node=1 \\
    -m peft.scripts.train \\
    --model_name_or_path "{self.config['base_model']}" \\
    --train_file "{training_data_path}" \\
    --output_dir "{adapter_output_dir}" \\
    --lora_r {self.config['r']} \\
    --lora_alpha {self.config['lora_alpha']} \\
    --lora_dropout {self.config['lora_dropout']} \\
    --target_modules {' '.join(self.config['target_modules'])} \\
    --num_train_epochs {self.config['epochs']} \\
    --learning_rate {self.config['learning_rate']} \\
    --per_device_train_batch_size {self.config['batch_size']} \\
    --gradient_accumulation_steps {self.config['gradient_accumulation_steps']} \\
    --warmup_ratio {self.config['warmup_ratio']} \\
    --save_steps {self.config['save_steps']} \\
    --logging_steps {self.config['logging_steps']} \\
    --fp16 \\
    --report_to none

echo "学習完了: {adapter_output_dir}"
'''

        script_path = self.scripts_dir / f"train_{output_name}.sh"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        logger.info(f"学習スクリプト生成: {script_path}")
        return script_path

    def generate_python_training_script(
        self,
        training_data_path: Path,
        output_name: Optional[str] = None
    ) -> Path:
        """
        Python学習スクリプトを生成（より詳細な制御が可能）

        Args:
            training_data_path: 学習データのパス
            output_name: 出力アダプター名

        Returns:
            生成したスクリプトのパス
        """
        if output_name is None:
            output_name = f"awareness_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        adapter_output_dir = self.output_dir / output_name

        script_content = f'''"""
気づき創発システム - LoRA学習スクリプト
生成日時: {datetime.now().isoformat()}
学習データ: {training_data_path}
"""

import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import torch

# 設定
BASE_MODEL = "{self.config['base_model']}"
TRAINING_DATA = "{training_data_path}"
OUTPUT_DIR = "{adapter_output_dir}"

LORA_CONFIG = {{
    "r": {self.config['r']},
    "lora_alpha": {self.config['lora_alpha']},
    "target_modules": {self.config['target_modules']},
    "lora_dropout": {self.config['lora_dropout']},
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM
}}

TRAINING_CONFIG = {{
    "num_train_epochs": {self.config['epochs']},
    "learning_rate": {self.config['learning_rate']},
    "per_device_train_batch_size": {self.config['batch_size']},
    "gradient_accumulation_steps": {self.config['gradient_accumulation_steps']},
    "warmup_ratio": {self.config['warmup_ratio']},
    "save_steps": {self.config['save_steps']},
    "logging_steps": {self.config['logging_steps']},
    "fp16": True,
    "output_dir": OUTPUT_DIR,
    "report_to": "none"
}}


def load_training_data(path: str) -> Dataset:
    """学習データを読み込む"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                # messages形式をテキストに変換
                messages = item.get("messages", [])
                text = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "user":
                        text += f"User: {{content}}\\n"
                    else:
                        text += f"Assistant: {{content}}\\n"
                data.append({{"text": text.strip()}})
            except json.JSONDecodeError:
                continue
    return Dataset.from_list(data)


def main():
    print(f"ベースモデル: {{BASE_MODEL}}")
    print(f"学習データ: {{TRAINING_DATA}}")
    print(f"出力先: {{OUTPUT_DIR}}")

    # トークナイザーとモデルの読み込み
    print("モデル読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA設定
    print("LoRA設定中...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # データ読み込み
    print("データ読み込み中...")
    dataset = load_training_data(TRAINING_DATA)
    print(f"学習データ数: {{len(dataset)}}")

    # トークナイズ
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # データコレーター
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 学習設定
    training_args = TrainingArguments(**TRAINING_CONFIG)

    # トレーナー
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # 学習実行
    print("学習開始...")
    trainer.train()

    # 保存
    print(f"モデル保存: {{OUTPUT_DIR}}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("学習完了!")


if __name__ == "__main__":
    main()
'''

        script_path = self.scripts_dir / f"train_{output_name}.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        logger.info(f"Python学習スクリプト生成: {script_path}")
        return script_path

    def get_available_adapters(self) -> list[dict]:
        """利用可能なアダプター一覧を取得"""
        adapters = []
        for adapter_dir in self.output_dir.iterdir():
            if adapter_dir.is_dir() and adapter_dir.name != "scripts":
                config_file = adapter_dir / "adapter_config.json"
                if config_file.exists():
                    with open(config_file, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    adapters.append({
                        "name": adapter_dir.name,
                        "path": str(adapter_dir),
                        "config": config
                    })
        return adapters

    def get_training_status(self) -> dict:
        """学習状況の概要を取得"""
        readiness = self.check_readiness()
        adapters = self.get_available_adapters()

        return {
            "readiness": readiness,
            "available_adapters": len(adapters),
            "adapters": adapters,
            "config": self.config
        }


class TrainingNotifier:
    """学習準備完了の通知"""

    def __init__(self, trainer: LoRATrainer):
        self.trainer = trainer
        self.last_notified_count = 0

    def check_and_notify(self) -> Optional[str]:
        """
        学習準備状況をチェックして通知メッセージを返す

        Returns:
            通知メッセージ（通知不要ならNone）
        """
        readiness = self.trainer.check_readiness()

        if readiness["ready"] and readiness["current_samples"] > self.last_notified_count:
            self.last_notified_count = readiness["current_samples"]
            return (
                f"LoRA学習の準備が整いました！\n"
                f"蓄積データ: {readiness['current_samples']}件\n"
                f"必要最小: {readiness['required_samples']}件\n"
                f"\n"
                f"学習を開始するには `!lora train` コマンドを実行してください。"
            )

        # 進捗通知（25%, 50%, 75%で通知）
        progress = readiness["progress_percent"]
        milestones = [25, 50, 75]
        for milestone in milestones:
            if progress >= milestone and self.last_notified_count < milestone:
                self.last_notified_count = milestone
                return (
                    f"学習データ蓄積進捗: {progress:.0f}%\n"
                    f"現在: {readiness['current_samples']}件 / "
                    f"目標: {readiness['required_samples']}件"
                )

        return None


# テスト用
if __name__ == "__main__":
    from awareness_database import AwarenessDatabase

    # テスト用データベース
    db = AwarenessDatabase(data_dir="./data/awareness_test")

    # トレーナー初期化
    trainer = LoRATrainer(db, output_dir="./data/lora_test")

    # 状況確認
    status = trainer.get_training_status()
    print(f"学習状況: {json.dumps(status, ensure_ascii=False, indent=2)}")

    # スクリプト生成テスト
    test_data_path = Path("./data/awareness_test/training_data.jsonl")
    if test_data_path.exists():
        script_path = trainer.generate_python_training_script(test_data_path)
        print(f"スクリプト生成: {script_path}")
