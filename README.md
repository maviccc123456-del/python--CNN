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
├── data/              # CIFAR-10データセット格納ディレクトリ（自動ダウンロード）
├── model/             # 学習済みモデル保存ディレクトリ（手動作成が必要）
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

### 事前準備

1. モデル保存ディレクトリの作成：
```bash
mkdir model
```

2. 初回実行時、CIFAR-10データセットが自動的に `./data` ディレクトリにダウンロードされます

### モード1：モデルの学習

**実行が必要なコード：**

`if __name__=='__main__':` 部分で、以下のコードのコメントを解除してください：

```python
if __name__=='__main__':
    # 1. データセットの取得（必須）
    train_dataset, test_dataset = create_dataset()
    
    # 2. モデルの学習（必須）
    train(train_dataset)
```

**オプションの補助コード：**

```python
# データセット情報の確認（オプション）
print(f'訓練セット：{train_dataset.data.shape}')  # (50000, 32, 32, 3)
print(f'テストセット：{test_dataset.data.shape}')  # (10000, 32, 32, 3)
print(f'データセットクラス：{train_dataset.class_to_idx}')

# 画像の表示（オプション）
plt.figure(figsize=(2,2))
plt.imshow(train_dataset.data[11])
plt.title(train_dataset.targets[11])
plt.show()

# モデルパラメータの確認（オプション）
model = ImageModel()
summary(model, (3, 32, 32), batch_size=1)
```

**コメントアウトが必要なコード：**

```python
# evaluate(test_dataset)  # 学習時はテストは不要
```

**学習パラメータの設定：**
- バッチサイズ (BATCH_SIZE): 8
- エポック数 (epochs): 10
- 学習率: 0.001
- オプティマイザ: Adam
- 損失関数: CrossEntropyLoss

**学習出力例：**
```
epoch:1, loss:1.23456, acc:0.45, time:120.50s
epoch:2, loss:1.10234, acc:0.52, time:118.30s
epoch:3, loss:0.98765, acc:0.58, time:119.20s
...
epoch:10, loss:0.65432, acc:0.72, time:118.80s
```

学習完了後、モデルは自動的に `./model/image_model.pth` に保存されます

---

### モード2：モデルの評価

**実行が必要なコード：**

`if __name__=='__main__':` 部分で、以下のコードのコメントを解除してください：

```python
if __name__=='__main__':
    # 1. データセットの取得（必須）
    train_dataset, test_dataset = create_dataset()
    
    # 2. モデルの評価（必須）
    evaluate(test_dataset)
```

**コメントアウトが必要なコード：**

```python
# 以下のコードは評価時には実行不要

# train(train_dataset)  # 学習は不要

# データセット情報の確認は不要
# print(f'訓練セット：{train_dataset.data.shape}')
# print(f'テストセット：{test_dataset.data.shape}')
# print(f'データセットクラス：{train_dataset.class_to_idx}')

# 画像表示は不要
# plt.figure(figsize=(2,2))
# plt.imshow(train_dataset.data[11])
# plt.title(train_dataset.targets[11])
# plt.show()

# モデルパラメータの確認は不要
# model = ImageModel()
# summary(model, (3, 32, 32), batch_size=1)
```

**評価出力例：**
```
ACC: 0.65
```

**注意事項：**
- 評価前に必ず学習を完了し、`./model/image_model.pth` ファイルが存在することを確認してください
- 評価モードでは、保存済みのモデルパラメータを読み込んで評価を行います
- 評価時にモデルパラメータは変更されません

---

## 完全なコード実行フロー

### 初回使用（学習）

```python
if __name__=='__main__':
    # データセットの取得
    train_dataset, test_dataset = create_dataset()
    
    # モデルの学習
    train(train_dataset)
```

### 2回目の使用（評価）

```python
if __name__=='__main__':
    # データセットの取得
    train_dataset, test_dataset = create_dataset()
    
    # モデルの評価
    evaluate(test_dataset)
```

## クイックデバッグのヒント

コードの動作を素早くテストしたい場合：

1. **エポック数を減らす**：`epochs=10` を `epochs=1` に変更
2. **各エポックで1バッチのみ学習**：train関数のバッチループの最後にある `break` のコメントを解除
3. **1エポックのみ学習**：train関数のエポックループの最後にある `break` のコメントを解除

## パフォーマンス最適化の提案

- エポック数を増やす（例：epochs=20 または 50）ことで精度を向上
- バッチサイズを調整（例：BATCH_SIZE=32 または 64）
- GPU加速を使用：
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  x, y = x.to(device), y.to(device)
  ```
- データ拡張を追加してモデルの汎化性能を向上

## ライセンス

本プロジェクトは学習・研究目的でのみ使用してください
