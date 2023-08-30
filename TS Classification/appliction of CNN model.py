from typing import Any

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from extraction_of_time_series import ts_extraction


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


def align(li, max_len):
    new_list = []
    for l in li:
        if len(l) < max_len:
            l = l + [0] * (max_len - len(l))
        new_list.append(l)
    return new_list


def uniform_sampling(lst, m):
    return [row[::m] for row in lst if max(row) > 50]


if __name__ == '__main__':
    date = r'2023-06-12*'
    module = ts_extraction(date)
    draw_data = module.run()
    max_cord = module.draw_pic()
    ts_data = [list(d[1]) for d in draw_data]
    ts_data = uniform_sampling(ts_data, 20)
    ts_data = align(ts_data, 491)
    input = torch.Tensor(ts_data)
    model = CNNnet()
    model.cuda()
    model.load_state_dict(torch.load('CNN Parameter.pth'))
    model.eval()
    with torch.no_grad():
        if len(input) > 0:
            input = input.to('cuda')
            num = model(input.reshape(input.shape[0], 1, -1))
            output = torch.argmax(num, dim=1).tolist()
            print(output)
            for i in range(len(draw_data)):
                plt.plot(draw_data[i][0], draw_data[i][1], label=f'Cluster:{output[i]+1}')
                plt.legend()
        else:
            plt.text(1100, max_cord, 'No Clusters Found', fontsize=15, c='red')
        plt.show()
    exit()
