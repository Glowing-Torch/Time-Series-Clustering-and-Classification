from typing import Any

import torch
import torch.nn as nn
from extraction_of_static_input import edge_extraction
import matplotlib.pyplot as plt


class BPModel(nn.Module):
    def __init__(self):
        super().__init__()
        #全连接层节点数
        #全连接层节点数
        self.layer1 = nn.Linear(3, 36)
        self.layer2 = nn.Linear(36, 48)
        self.layer3 = nn.Linear(48, 24)
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


if __name__=="__main__":
    file_name = r'E:\data\1.5y data\2023-06-29_00-24-59.csv'
    data = edge_extraction(file_name)
    input,draw_part = data.run()
    max_cord=data.plot_chart()
    model=BPModel()
    model.load_state_dict(torch.load('best_True'))
    model.eval()
    with torch.no_grad():
        if len(input)>0:
            input=torch.Tensor(input)
            num=model(input)
            output = torch.argmax(num, dim=1).tolist()
            for i in range(len(draw_part)):
                plt.plot(draw_part[i][0],draw_part[i][1],label=f'Cluster:{output[i]}')
                plt.legend()
        else:
            plt.text(1100,max_cord,'No Clusters Found',fontsize=15,c='red')
        plt.show()


