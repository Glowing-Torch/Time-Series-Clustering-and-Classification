import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score


# 输入数据的容器，配合DataLoader，完成mini_bacth训练方法，要包含__init__， __len__和__getitem__三个属性
class MyDataset(Dataset):
    def __init__(self, train_lines):
        super().__init__()
        train_lines = train_lines.dropna(axis=0, how='any')
        self.lens = train_lines.shape[0]  # 数据量
        self.x_data = []
        self.y_data = []
        cop_train=train_lines.loc[train_lines['Cluster']!=-1,:].copy()
        cop_train = cop_train.reset_index(drop=True)
        self.lens=cop_train.shape[0]

        # 读取数据部分
        print(self.lens)

        for i in range(self.lens):

            x = cop_train.loc[i,['Wirkleistung/W','Blindleistung/Var','Volatile']].copy().to_list()  # 输入特征
            y = cop_train.loc[i,'Cluster'].item()  # 标签值
            self.x_data.append(x)
            self.y_data.append([y])

    # 数据总数
    def __len__(self):
        return self.lens

    # 根据下标获取其中一条数据
    def __getitem__(self, index):
        # 转换为网络输入形式
        x_data = torch.Tensor(list(map(float, self.x_data[index])))
        y_data = torch.squeeze(torch.Tensor(list(map(float, self.y_data[index]))))
        return x_data, y_data.long()

# BP神经网络结构，注意BatchNorm的使用
class BPModel(nn.Module):
    def __init__(self):
        super().__init__()
        #全连接层节点数
        self.layer1 = nn.Linear(3, 36)
        self.layer2 = nn.Linear(36, 48)
        self.layer3 = nn.Linear(48, 24)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.BN0 = nn.BatchNorm1d(3, momentum=0.5)
        self.BN1 = nn.BatchNorm1d(36, momentum=0.5)
        self.BN2 = nn.BatchNorm1d(48, momentum=0.5)

    def forward(self, x):
        #可自行添加dropout和BatchNorm1d层
        x = self.BN0(x)
        x = self.BN1(self.layer1(x))
        x = torch.relu(x)
        x = self.BN2(self.layer2(x))
        x = torch.relu(x)
        out = torch.relu(self.layer3(x))
        return out

class BPTrain(object):
    # train_path为训练集文件，val_path为验证集文件
    def __init__(self, train_path, val_path, lr=0.2, epochs=500, gpu=False):
        self.gpu = gpu  # 是否选用gpu， 默认为False
        self.lr = lr  # 学习率
        self.epochs = epochs  # 迭代次数
        self.loss = []  # 用于绘制loss图像
        self.num_epoch = []
        self.recall_micro=[]

        # 训练集
        training_sets = pd.read_excel(train_path,
                                      engine='openpyxl')
        train_dataset = MyDataset(training_sets)
        self.gen = DataLoader(train_dataset, shuffle=True, batch_size=256, drop_last=True)

        # 测试集
        testing_sets=pd.read_excel(val_path,
                      engine='openpyxl')
        val_dataset = MyDataset(testing_sets)
        self.val_gen = DataLoader(val_dataset, batch_size=300, shuffle=False)

    # 权重初始化（好的初始化有利于模型训练）
    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):

            torch.nn.init.kaiming_uniform_(
                m.weight.data,
                a=0,
                mode='fan_in',
                nonlinearity='leaky_relu')

            m.bias.data.zero_()

    # model_path为模型参数文件
    def train(self, model_path=None):
        # 设备选择
        if not self.gpu:
            device = 'cpu'
        else:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print(f'gpu is unavailable !!!')
                device = 'cpu'

        # 网络实例化，并设计损失函数、optimizer和lr衰减
        best_val_rec = 0
        model = BPModel()
        if model_path:
            model.load_state_dict(torch.load(model_path))
        else:
            model.apply(self.weights_init)
        model = model.to(device)

        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 300, gamma=0.9)

        for epoch in range(self.epochs):
            all_loss = 0
            train_rights = 0
            train_falses = 0
            for i, data in enumerate(self.gen, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                model.train()
                optimizer.zero_grad()  # 梯度清零

                output = model(inputs)
                loss = loss_func(output, labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                output = torch.argmax(output, dim=1)
                train_count_right = output == labels  # mini_batch中正确项
                train_count_false = output != labels  # mini_batch中错误项
                train_rights += sum(train_count_right)
                train_falses += sum(train_count_false)
                all_loss += loss

            self.loss.append(float(all_loss))
            self.num_epoch.append(epoch)

            # 验证集检验频率
            if epoch % 10 == 0:
                lr=optimizer.state_dict()['param_groups'][0]['lr']
                recall_micro = recall_score(labels.to('cpu'), output.to('cpu'), average='micro')  # 微平均
                print('\n')
                print(f'number of iteration:{epoch},loss:{all_loss},learning rate:{lr}')
                print(f'recall in training sets:{recall_micro}')
                model.eval()  # model.eval() 会关闭BN和dropout
                with torch.no_grad():
                    for j, data in enumerate(self.val_gen, 0):
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        output = model(inputs)
                        output = torch.argmax(output, dim=1)
                        recall_micro = recall_score(labels.to('cpu'), output.to('cpu'), average='micro')
                        self.recall_micro.append(recall_micro)# 微平均
                        # print('recall in testing sets:',recall_micro)

                    if (recall_micro >= 0.95) &(recall_micro>=best_val_rec):
                        best_val_rec = recall_micro
                        input_tuple = torch.randn(32, inputs.shape[1]).to(device)
                        torch.onnx.export(
                            model,
                            input_tuple,
                            'BP_model.onnx',
                            export_params=True,
                            opset_version=8,
                        )
                        torch.save(model.state_dict(), f'best_{self.gpu}')
                        print('recall in testing sets:',best_val_rec)


    # 绘制Loss值图像
    def draw_loss(self):
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(2, 1, 1)

        ax1.set_xlabel('epoch')
        ax1.set_ylabel('recall')
        ax1.set_title("Recall Score Chart")
        a=self.num_epoch[::10]
        # print(a,self.recall_micro)
        ax1.plot(a, self.recall_micro)
        ax2 = fig.add_subplot(2, 1, 2)

        ax2.set_xlabel('epoch')
        ax2.set_ylabel('loss')
        ax2.set_title("Loss Chart")

        ax2.plot(self.num_epoch, self.loss)
        # plt.savefig('Loss_chart.jpg')
        plt.show()


if __name__ == '__main__':
    # 按照自己文件名
    train_path = r'D:\PyCharm 2020.1.3\MyProgram\praktikum\data with label 2.xls'
    val_path = r'D:\PyCharm 2020.1.3\MyProgram\praktikum\test data.xlsx'
    model_path = r'best_False'  # 训练好的权重文件
    pre_path = 'predata.txt'  # 预测数据

    training_BP = BPTrain(train_path, val_path,gpu=True,epochs=3000)
    training_BP.train(model_path=None)
    training_BP.draw_loss()


