import torch
import pandas as pd
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score

def find_max(li):
    len_list=[]
    for l in li:
        len_list.append(len(l))
    return max(len_list)


def align(li,max_len):
    new_list=[]
    for l in li:
        if len(l)<max_len:
            l=l+[0]*(max_len-len(l))
        new_list.append(l)
    return new_list

def get_your_data(train_path):
    df=pd.read_excel(train_path,engine='openpyxl')
    train_lines = df.dropna(axis=0, how='any')
    cop_train=train_lines
    x_data=[]
    y_data=[]
    cop_train = cop_train.reset_index(drop=True)
    lens = cop_train.shape[0]
    for i in range(lens):
        x = eval(cop_train.loc[i, 'value'])  # 输入特征
        y = cop_train.loc[i, 'new_label'].item()  # 标签值
        x_data.append(x)
        y_data.append(y)
    max_length = find_max(x_data)
    x_data = align(x_data, max_length)
    return x_data,y_data


class MyDataset(Dataset):
    def __init__(self, data_features,data_target):
        super().__init__()

        self.lens = len(data_features)  # 数据量
        self.x_data =data_features
        self.y_data=data_target
        self.x_data = torch.Tensor(self.x_data)
        self.y_data = torch.squeeze(torch.Tensor(list(map(float, self.y_data))))
        # 读取数据部分


    # 数据总数
    def __len__(self):
        return self.lens

    # 根据下标获取其中一条数据
    def __getitem__(self, index):
        # 转换为网络输入形式
        return self.x_data[index], self.y_data[index]


class CNNnet(nn.Module):
    def __init__(self, *, inputLength=491, kernelSize=3, kindsOutput=26):
        super().__init__()
        filterNum1 = 32
        filterNum2 = 64
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, filterNum1, kernelSize),  # inputLength - kernelSize + 1 = 491 - 3 + 1 = 489
            nn.BatchNorm1d(filterNum1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernelSize, stride=1)  # 489 - 3 + 1 = 487
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(filterNum1, filterNum2, kernelSize),  # 487 - 3 + 1 = 485
            nn.BatchNorm1d(filterNum2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernelSize, stride=1)  # 485 - 3 + 1 = 483
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(filterNum2 * (inputLength - 8), kindsOutput)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropout(x)
        return x


def datasetSplit(data,label):
    train_data=data[:3480]
    train_label=label[:3480]
    test_data=data[3480:]
    test_label=label[3480:]
    test_label=torch.tensor(test_label)
    test_data=torch.tensor(test_data)
    train_data=torch.tensor(train_data)
    train_label=torch.tensor(train_label)
    return train_data,train_label,test_data,test_label


def train(trainData, trainLabel, *, savePath='..\models\pytorch', modelName = 'model.pt', epochs = 100, batchSize = 256, classNum = 26):

    trainFeatures, trainTarget, testFeatures, testTarget = datasetSplit(trainData, trainLabel)
    print('trainFeatures shape:', trainFeatures.shape, '\ttestFeatures shape:', testFeatures.shape)
    trainSet = MyDataset(trainFeatures, trainTarget)
    trainLoader = DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True, drop_last=True)

    model = CNNnet(inputLength=trainFeatures.shape[1], kindsOutput = classNum)
    # model.load_state_dict(torch.load('CNN Parameter.pth'))
    model=model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.7)
    model.train()
    start_time = time.time()
    loss_list=[]
    num_epoche=[]
    score_list=[]
    for epoch in range(epochs):
        all_loss=0
        for seq, y_train in trainLoader:
            seq = seq.to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()
            y_pred = model(seq.reshape(batchSize, 1, -1))
            y_train = y_train.long()
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            all_loss+=loss

        loss_list.append(float(all_loss))
        num_epoche.append(epoch)
        # compute test accuracy
        model.eval()
        testFeatures=testFeatures.to(device)
        _y_pred = model(testFeatures.reshape(testFeatures.shape[0], 1, -1))
        y_pred = torch.max(_y_pred, 1)[1]
        y_pred=y_pred.cpu()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        recall_micro=recall_score(testTarget,y_pred,average='micro')
        score_list.append(recall_micro)
        print(f'Epoch: \t{epoch+1} \t Recall: {recall_micro:.3f} \t Loss: {loss.item():.5f} ,learning rate:{lr:.4f}'.replace(" ",""))
        # if recall_micro>0.96:
        #     torch.save(model.parameters(),'CNN Network.pth')
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    inputs = torch.randn(5, 1, trainFeatures.shape[1]).to(device)
    torch.onnx.export(
        model,
        inputs,
        'model_CNN.onnx',
        export_params=True,
        # opset_version=8,
    )
    draw_loss(loss_list,score_list,num_epoche)
    return model


def draw_loss(loss,score,epoch):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('recall')
    ax1.set_title("Recall Score Chart")
    # print(a,self.recall_micro)
    ax1.plot(epoch, score)
    ax2 = fig.add_subplot(2, 1, 2)

    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_title("Loss Chart")

    ax2.plot(epoch, loss)
    # plt.savefig('Loss_chart.jpg')
    plt.show()


if __name__=='__main__':
    train_path=r'final time series.xlsx'
    train_data,train_label = get_your_data(train_path)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    train(train_data,train_label)
