import torch
import torch. nn as nn
from torchvision.datasets import CIFAR10
from torchvision. transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

#每批次变量
BATCH_SIZE=8

#1准备数据集
def create_dataset():
    #1获取数据集
    # 参1：数据集路径，参2：是否是训练集，参3：数据预处理→张量数据，参4：是否联网下载
    train_dataset=CIFAR10(root='./data',train=True,transform=ToTensor(),download=True)
    #2获取数据集
    test_dataset=CIFAR10(root='./data',train=False,transform=ToTensor(),download=True)
    #3返回数据集
    return train_dataset,test_dataset

#2搭建神经网络
class ImageModel(nn.Module):
    #1.初始化父类成员，搭建神经网络
    def __init__(self):
        #1.1初始化父类成员
        super().__init__()
        #1.2搭建神经网络
        #第一卷积层。(输入3通道，输出6通道，卷积核大小3，步长1，填充0)
        self.conv1=nn.Conv2d(3,6,3,1,0)
        #第一个池化层窗口大小2x2，步长为2，填充为0
        self.pool1=nn.MaxPool2d(2,2,0)

        #第二个卷积层，输入6通道，输出16通道
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        #第二个池化层窗口大小2x2，步长为2，填充为0
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        #第一个隐藏层（全连接层）输入576，输出120
        self.linear1=nn.Linear(576,120)
        #第二个隐藏层（全连接层）输入576，输出120
        self.linear2=nn.Linear(120, 84)
        #第三个隐藏层（全连接层）|（输出层）|输入84，输出10
        self.output=nn.Linear(84, 10)
    def forward(self,x):
        #第一层：卷积层（加权求和）+激励层（激活函数）+池化层（降维）
        x=self.pool1(torch.relu(self.conv1(x)))

        # 第二层：卷积层（加权求和）+激励层（激活函数）+池化层（降维）
        x = self.pool2(torch.relu(self.conv2(x)))

        #全连接层只能处理2维数据，将数据拉平
        #参1：样本数（行数），参2：列数（特征数），-1：自动计算
        x=x.reshape(x.size(0),-1)  #8行576列
        # print(f'x.shape:{x.shape}')

        #第3层：全连接层（加权求和）+激励层（激活函数）
        x=torch.relu(self.linear1(x))

        #第4层：全连接层（加权求和）+激励层（激活函数）
        x = torch.relu(self.linear2(x))

        #第5层（加权求和）→输出层
        return self.output(x)


#3模型训练
def train(train_dataset):
    #1.创建数据加载器。
    dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    #2.创建模型对象。
    model=ImageModel()
    #3创建损失函数对象。
    criterion=nn.CrossEntropyLoss()  #多分类交叉熵损失函数= softmax()激活函数+损失计算。
    #4.创建优化器对象
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    #5.遍历循环epoch，开始每轮的调练动作
    #5.1定义变量，记录调练的总轮数
    epochs=10
    #5.2遍历，完成每轮的 所有批次的调练动作
    for epoch_idx in range(epochs):
        #5.2.1定义变量，记录：总损失，总样本数据量，预测正确样本个数，训练（开始）时间
        total_loss,total_sample,total_correct,start=0.0,0,0,time.time()
        #5.2.2 遍历数据加载器，获取到每批次的数据。
        for x,y in dataloader:
            #5.2.3切换训练模式
            model.train()
            #5.2.4模型预测
            y_pred=model(x)
            #5.2.5计算损失
            loss=criterion(y_pred,y)
            #5.2.6梯度清零+反向传播+参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            #5.2.7预测正确的样本个数
            # print(y_pred)# 批次中，每张图 每个分类的预测概率

            # argmax()返回最大值对应索引，充当该图片的预测分类
            #tensor([9,8,5,5,1,5,8,5])
            # print(torch.argmax(y_pred,dim=-1))# -1表示行 ,预测分类
            # print(y)                          # 预测分类
            # print(torch.argmax(y_pred,dim=1)==y) #是否预测正确
            # print((torch.argmax(y_pred, dim=1) == y).sum()) #预测正确的样本个数
            total_correct+=(torch.argmax(y_pred, dim=1) == y).sum()

            #5.2.8统计当前批次的总损失         第一批的总损失*第一批的样本个数
            total_loss+=loss.item()*len(y) #[一批的总损失+第二批的总损失+第三批+……]

            #5.2.9
            total_sample+=len(y)
            #break 每轮只训练1批，提高训练效率

        #5.2.10
        print(f'epoch:{epoch_idx+1},loss:{total_loss/total_sample:.5f},acc{total_correct/total_sample:.2f},time:{time.time()-start:.2f}s')
        #break #这里写break意味着只训练一轮

    #6保存模型
    torch.save(model.state_dict(),f'./model/image_model.pth')
#4模型测试
def evaluate(test_dataset):
    #1.创建测试集，数据加载器
    dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    #2.创建模型对象
    model=ImageModel()
    #3.加载模型参数
    model.load_state_dict(torch.load('./model/image_model.pth')) #picle文件
    #4.定义变量统计，预测正确样本个数，总样本个数
    total_correct,total_samples=0,0
    #5.遍历数据加载器，获取到每批次的数据
    for x,y in dataloader:
        #5.1切换模型模式
        model.eval()
        #5.2 模型预测
        y_pred=model(x)
        #5.3因为训练的适合用了CrossEntropyloss，所有搭建神经网络时没有加softmax（）激活函数，这里用argmax（来模拟）
        #argmax()函数功能：返回最大值对应的索引，充当→该图片的分类，预测分类
        y_pred=torch.argmax(y_pred,dim=-1)#-1这里表示行
        #5.4统计预测正确的样本个数。
        total_correct+=(y_pred==y).sum()
        #5.5统计总样本个数
        total_samples+=len(y)

    #6.打印正确率（预测结果）
    print(f'ACC:{total_correct/total_samples:.2f}')
#5测试
if __name__=='__main__':
    # #获取数据集
    train_dataset, test_dataset = create_dataset()
    # print(f'训练集：{train_dataset.data.shape}')  #（50000，32，32，3）
    # print(f'测试集：{train_dataset.data.shape}')  #（10000，32，32，3）
    # # {airplane:0 | automobile:1 | bird:2 | cat:3 | deer:4 | dog:5 | frog:6 | horse:7 | ship:8 | truck:9}
    # print(f'数据集类别：{train_dataset.class_to_idx}')
    #
    # # 图像展示
    # plt.figure(figsize=(2,2))
    # plt.imshow(train_dataset.data[11])
    # plt.title(train_dataset.targets[11])
    # plt.show()

    #2.搭建神经网络
    # model=ImageModel()
    # #查看模型参数
    # summary(model,(3,32,32),batch_size=1)

    # #3.模型训练
    # train(train_dataset)
    #4.模型测试
    evaluate(test_dataset)