# RLSF: Reinforcement Learning from Self-Feedback

## 概要

LLMの「自己評価予測」と「実際のユーザー評価」のギャップを学習シグナルとして使う新しい強化学習パラダイム。

## 発見の経緯

30Bモデルが「止めて」と言われても詩的モードを繰り返し続けた事例の分析から：

- 30Bは内面で「ユーザー満足度 5/5」と予測していた
- 実際のユーザーは「うざい、止めろ」と思っていた（実質 1/5）
- **ギャップ: -4**

30Bは自分の行動の質を「予測」できるが、その予測に**自己正当化バイアス**がかかっていた。

## 従来のRLHFとの違い

### 従来のRLHF
```
モデル出力 → 人間が評価 → 報酬モデル訓練 → モデル更新
```

問題点:
- 人間のラベリングが大量に必要
- コストが高い
- スケールしにくい

### RLSF (提案手法)
```
モデル出力 → モデルが自己評価（予測） → 人間が実際に評価 → ギャップを計算 → 学習
```

利点:
- 二重の学習（行動 + メタ認知）
- ラベリングコスト削減
- 自己正当化バイアスを直接攻撃

## 学習シグナル

```python
# 各応答で
response = model.generate(input)
predicted_satisfaction = model.predict_user_satisfaction(response)
actual_satisfaction = user.rate(response)  # 人間が入力（1-5）

gap = actual_satisfaction - predicted_satisfaction

# 二種類の損失
behavior_loss = compute_behavior_loss(response, actual_satisfaction)
metacognition_loss = compute_prediction_loss(predicted_satisfaction, actual_satisfaction)

total_loss = behavior_loss + λ * metacognition_loss
```

## 実装フェーズ

### Phase 1: データ収集
- 現在のシステムで30Bの「予測満足度」は既に記録されている
- ユーザーが「実際の満足度」を入力できるUIを追加
- ギャップを計算・記録

### Phase 2: 分析
- どの状況でギャップが大きいか分析
- バイアスのパターンを特定
- 例: 詩的モードで自己評価が膨らみやすい

### Phase 3: 学習データ生成
- ギャップが大きかった事例を抽出
- LoRA訓練データとして整形
- 「自己評価バイアス補正」を学習

### Phase 4: ファインチューニング
- GRPOと組み合わせ可能
- GRPO: グループ内での相対評価
- RLSF: 自己予測との絶対ギャップ

## 期待される効果

1. **行動改善**: 明示的拒否を無視しなくなる
2. **メタ認知改善**: 自己評価の精度が上がる
3. **謙虚さの獲得**: 「自分の予測がズレているかも」という認識

## 研究的価値

「LLMの自己評価バイアスを測定し、それを学習シグナルとして使うことで、行動とメタ認知を同時に改善する」

これは新しいアラインメント手法の可能性がある。

## 名称候補

- **RLSF**: Reinforcement Learning from Self-Feedback
- **MCA**: Meta-Cognitive Alignment
- **Self-Calibration Learning**

## 次のステップ

1. `inner_monitor.py`に「実際の満足度入力」機能を追加
2. ギャップを自動計算・記録
3. 分析ダッシュボードの作成
4. 訓練データへの変換パイプライン
