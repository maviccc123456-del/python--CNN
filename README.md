# CIFAR-10 画像分類プロジェクト

PyTorchを使用したCIFAR-10データセットの画像分類モデル。畳み込みニューラルネットワーク(CNN)による学習と評価を実装しています。

## プロジェクト概要

本プロジェクトは、CIFAR-10データセットを用いた画像分類システムを実装しています。CIFAR-10は10クラスの60,000枚の32x32カラー画像で構成され、50,000枚の訓練画像と10,000枚のテスト画像に分かれています。

### データセットのクラス

- airplane (飛行機) - 0
- automobile (自動車) - 1
- bird (鳥) - 2
- cat (猫) - 3
- deer (鹿) - 4
- dog (犬) - 5
- frog (カエル) - 6
- horse (馬) - 7
- ship (船) - 8
- truck (トラック) - 9

## 環境要件

```bash
pip install torch torchvision matplotlib torchsummary
```

**主な依存ライブラリ：**
- PyTorch
- torchvision
- matplotlib
- torchsummary

## プロジェクト構造

```
.
├── data/              # CIFAR-10データセット格納ディレクトリ
├── model/             # 学習済みモデル保存ディレクトリ
└── main.py            # メインプログラムファイル
```

## モデルアーキテクチャ

本プロジェクトで使用するCNNモデルは以下の層で構成されています：

1. **第1畳み込みブロック**
   - 畳み込み層：3 → 6チャネル、カーネルサイズ 3x3
   - ReLU活性化関数
   - 最大プーリング層：2x2

2. **第2畳み込みブロック**
   - 畳み込み層：6 → 16チャネル、カーネルサイズ 3x3
   - ReLU活性化関数
   - 最大プーリング層：2x2

3. **全結合層**
   - FC1: 576 → 120
   - FC2: 120 → 84
   - 出力層: 84 → 10

## 使用方法

### 1. データの準備

初回実行時、プログラムは自動的にCIFAR-10データセットを `./data` ディレクトリにダウンロードします。

### 2. モデルの学習

コード内の学習部分のコメントを解除してください：

```python
if __name__=='__main__':
    train_dataset, test_dataset = create_dataset()
    train(train_dataset)
```

**学習パラメータ：**
- バッチサイズ (BATCH_SIZE): 8
- エポック数 (epochs): 10
- 学習率: 0.001
- オプティマイザ: Adam
- 損失関数: CrossEntropyLoss

### 3. モデルの評価

評価関数を実行してください：

```python
if __name__=='__main__':
    train_dataset, test_dataset = create_dataset()
    evaluate(test_dataset)
```

### 4. データの可視化

可視化コードのコメントを解除してデータセットのサンプルを表示できます：

```python
plt.figure(figsize=(2,2))
plt.imshow(train_dataset.data[11])
plt.title(train_dataset.targets[11])
plt.show()
```

### 5. モデル構造の確認

```python
model = ImageModel()
summary(model, (3, 32, 32), batch_size=1)
```

## 学習出力例

```
epoch:1, loss:1.23456, acc:0.45, time:120.50s
epoch:2, loss:1.10234, acc:0.52, time:118.30s
...
```

## 評価出力例

```
ACC: 0.65
```

## モデルの保存

学習完了後、モデルのパラメータは自動的に `./model/image_model.pth` に保存されます。

## 注意事項

1. 初回実行時はインターネット接続が必要です（CIFAR-10データセットのダウンロード）
2. モデル保存用の `model` ディレクトリを作成してください
3. `BATCH_SIZE` と `epochs` を変更することで学習パラメータを調整できます
4. 高速テストを行う場合は、コード内の `break` 文のコメントを解除してください

## パフォーマンス最適化の提案

- エポック数を増やして精度を向上させる
- 学習率を調整してハイパーパラメータを最適化する
- GPU加速を使用する（`.cuda()` または `.to(device)` を追加）
- データ拡張技術を追加してモデルの汎化性能を向上させる

## 開発者

PyTorch深層学習フレームワークを使用して実装

## ライセンス

本プロジェクトは学習・研究目的でのみ使用してください
