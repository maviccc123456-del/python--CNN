import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# バッチサイズ変数
BATCH_SIZE = 8

# 1. データセットの準備
def create_dataset():
    # 1. データセットの取得
    # 引数1：データセットのパス、引数2：訓練セットかどうか、引数3：データ前処理→テンソルデータ、引数4：オンラインダウンロードするかどうか
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    # 2. データセットの取得
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    # 3. データセットの返却
    return train_dataset, test_dataset

# 2. ニューラルネットワークの構築
class ImageModel(nn.Module):
    # 1. 親クラスのメンバーを初期化し、ニューラルネットワークを構築
    def __init__(self):
        # 1.1 親クラスのメンバーを初期化
        super().__init__()
        # 1.2 ニューラルネットワークの構築
        # 第1畳み込み層（入力3チャネル、出力6チャネル、カーネルサイズ3、ストライド1、パディング0）
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        # 第1プーリング層 ウィンドウサイズ2x2、ストライド2、パディング0
        self.pool1 = nn.MaxPool2d(2, 2, 0)

        # 第2畳み込み層、入力6チャネル、出力16チャネル
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        # 第2プーリング層 ウィンドウサイズ2x2、ストライド2、パディング0
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        # 第1隠れ層（全結合層）入力576、出力120
        self.linear1 = nn.Linear(576, 120)
        # 第2隠れ層（全結合層）入力120、出力84
        self.linear2 = nn.Linear(120, 84)
        # 第3隠れ層（全結合層）|（出力層）|入力84、出力10
        self.output = nn.Linear(84, 10)
    
    def forward(self, x):
        # 第1層：畳み込み層（重み付き和）+ 活性化層（活性化関数）+ プーリング層（次元削減）
        x = self.pool1(torch.relu(self.conv1(x)))

        # 第2層：畳み込み層（重み付き和）+ 活性化層（活性化関数）+ プーリング層（次元削減）
        x = self.pool2(torch.relu(self.conv2(x)))

        # 全結合層は2次元データのみ処理可能、データを平坦化
        # 引数1：サンプル数（行数）、引数2：列数（特徴数）、-1：自動計算
        x = x.reshape(x.size(0), -1)  # 8行576列
        # print(f'x.shape:{x.shape}')

        # 第3層：全結合層（重み付き和）+ 活性化層（活性化関数）
        x = torch.relu(self.linear1(x))

        # 第4層：全結合層（重み付き和）+ 活性化層（活性化関数）
        x = torch.relu(self.linear2(x))

        # 第5層（重み付き和）→ 出力層
        return self.output(x)


# 3. モデルの学習
def train(train_dataset):
    # 1. データローダーの作成
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 2. モデルオブジェクトの作成
    model = ImageModel()
    # 3. 損失関数オブジェクトの作成
    criterion = nn.CrossEntropyLoss()  # 多クラス交差エントロピー損失関数 = softmax()活性化関数 + 損失計算
    # 4. オプティマイザオブジェクトの作成
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # 5. エポックのループを回し、各ラウンドの学習動作を開始
    # 5.1 変数を定義し、学習の総ラウンド数を記録
    epochs = 10
    # 5.2 ループを回し、各ラウンドの全バッチの学習動作を完了
    for epoch_idx in range(epochs):
        # 5.2.1 変数を定義し、記録：総損失、総サンプル数、予測が正しいサンプル数、学習（開始）時間
        total_loss, total_sample, total_correct, start = 0.0, 0, 0, time.time()
        # 5.2.2 データローダーをループし、各バッチのデータを取得
        for x, y in dataloader:
            # 5.2.3 学習モードに切り替え
            model.train()
            # 5.2.4 モデル予測
            y_pred = model(x)
            # 5.2.5 損失の計算
            loss = criterion(y_pred, y)
            # 5.2.6 勾配をゼロにリセット + 逆伝播 + パラメータ更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5.2.7 予測が正しいサンプル数
            # print(y_pred)  # バッチ内の各画像の各分類の予測確率

            # argmax()は最大値に対応するインデックスを返し、その画像の予測分類とする
            # tensor([9,8,5,5,1,5,8,5])
            # print(torch.argmax(y_pred,dim=-1))  # -1は行を表す、予測分類
            # print(y)                             # 予測分類
            # print(torch.argmax(y_pred,dim=1)==y) # 予測が正しいかどうか
            # print((torch.argmax(y_pred, dim=1) == y).sum())  # 予測が正しいサンプル数
            total_correct += (torch.argmax(y_pred, dim=1) == y).sum()

            # 5.2.8 現在のバッチの総損失を統計    第1バッチの総損失 * 第1バッチのサンプル数
            total_loss += loss.item() * len(y)  # [1バッチの総損失 + 2バッチの総損失 + 3バッチ + ……]

            # 5.2.9
            total_sample += len(y)
            # break  各ラウンドで1バッチのみ学習し、学習効率を向上

        # 5.2.10
        print(f'epoch:{epoch_idx+1}, loss:{total_loss/total_sample:.5f}, acc:{total_correct/total_sample:.2f}, time:{time.time()-start:.2f}s')
        # break  ここにbreakを書くと1ラウンドのみ学習することを意味する

    # 6. モデルの保存
    torch.save(model.state_dict(), f'./model/image_model.pth')

# 4. モデルのテスト
def evaluate(test_dataset):
    # 1. テストセット、データローダーの作成
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # 2. モデルオブジェクトの作成
    model = ImageModel()
    # 3. モデルパラメータのロード
    model.load_state_dict(torch.load('./model/image_model.pth'))  # pickleファイル
    # 4. 変数を定義し統計、予測が正しいサンプル数、総サンプル数
    total_correct, total_samples = 0, 0
    # 5. データローダーをループし、各バッチのデータを取得
    for x, y in dataloader:
        # 5.1 モデルモードの切り替え
        model.eval()
        # 5.2 モデル予測
        y_pred = model(x)
        # 5.3 学習時にCrossEntropyLossを使用したため、ニューラルネットワーク構築時にsoftmax()活性化関数を追加していない、ここでargmax()でシミュレート
        # argmax()関数の機能：最大値に対応するインデックスを返し、その画像の分類、予測分類とする
        y_pred = torch.argmax(y_pred, dim=-1)  # -1はここでは行を表す
        # 5.4 予測が正しいサンプル数を統計
        total_correct += (y_pred == y).sum()
        # 5.5 総サンプル数を統計
        total_samples += len(y)

    # 6. 正解率を出力（予測結果）
    print(f'ACC:{total_correct/total_samples:.2f}')

# 5. テスト
if __name__ == '__main__':
    # データセットの取得
    train_dataset, test_dataset = create_dataset()
    # print(f'訓練セット：{train_dataset.data.shape}')  # (50000, 32, 32, 3)
    # print(f'テストセット：{test_dataset.data.shape}')  # (10000, 32, 32, 3)
    # # {airplane:0 | automobile:1 | bird:2 | cat:3 | deer:4 | dog:5 | frog:6 | horse:7 | ship:8 | truck:9}
    # print(f'データセットクラス：{train_dataset.class_to_idx}')
    #
    # # 画像表示
    # plt.figure(figsize=(2,2))
    # plt.imshow(train_dataset.data[11])
    # plt.title(train_dataset.targets[11])
    # plt.show()

    # 2. ニューラルネットワークの構築
    # model = ImageModel()
    # # モデルパラメータの確認
    # summary(model, (3, 32, 32), batch_size=1)

    # # 3. モデルの学習
    # train(train_dataset)
    # 4. モデルのテスト
    evaluate(test_dataset)
