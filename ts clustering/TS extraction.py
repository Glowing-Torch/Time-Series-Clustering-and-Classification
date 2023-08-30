"""
区分上升期与下降期：
如果两个幅值相似取上升期的起点与下降期的终点；
如果相差较大，舍弃
如果不连续，舍弃
"""

import glob
import os
from itertools import product
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from multiprocessing import Pool, Manager

# plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# 坐标轴负号的处理
plt.rcParams['axes.unicode_minus'] = False


def draw_figure(num, phase, files, wtr_list):
    global wrt_list
    ts_list = []
    dataset = []
    for file in files[(num - 1) * 24:num * 24]:
        df = pd.read_csv(file, delimiter=';')
        dataset.append(df)
    df = pd.concat(dataset)
    ph = f'p{phase}_power_active:W'
    y1 = np.array(df[ph])
    x = np.arange(1, df.shape[0] + 1)
    # fig = plt.figure()
    # ax2 = fig.add_subplot(211)
    # ax2.set(title='power_active', ylabel='power_active/W', xlabel='seconds/s')
    # ax2.plot(x, y1, color='blue')
    for i in range(1, len(y1) - 1):

        if (y1[i] - y1[i - 1] > 20) and (y1[i] - y1[i + 1] > 20):
            y1[i] = y1[i - 1]

    y1_smooth = savgol_filter(y1, 59, 1)
    # ax = fig.add_subplot(212)
    # ax.set(title='power_active_smooth',
    #        ylabel='power_active_smooth/W', xlabel='seconds/s')
    # ax.plot(x, y1_smooth, c='y')
    first_peaks = []
    for i in range(1, len(y1_smooth) - 1):
        if (y1_smooth[i] < y1_smooth[i - 1]) and (y1_smooth[i] < y1_smooth[i + 1]):
            first_peaks.append(i)
        elif (y1_smooth[i] > y1_smooth[i - 1]) and (y1_smooth[i] > y1_smooth[i + 1]):
            first_peaks.append(i)

    index_max_list = []
    index_min_list = []

    for i in range(3):
        disorder_list = []
        for i in range(1, len(first_peaks) - 1):
            delta1 = y1_smooth[first_peaks[i]] - y1_smooth[first_peaks[i - 1]]
            delta2 = y1_smooth[first_peaks[i]] - y1_smooth[first_peaks[i + 1]]
            if delta1 < 0 and delta2 < 0:
                if delta1 > -10 or delta2 > -10:
                    disorder_list.append(i)
        if not disorder_list:
            break
        else:
            j = 1
            for disorder in disorder_list:
                j -= 1
                del first_peaks[disorder + j]
    peaks = []
    peaks.append(first_peaks[0])
    values_list = []
    for peak in first_peaks:
        values_list.append(int(y1_smooth[peak]))

    for i in range(1, len(first_peaks) - 1):
        if (y1_smooth[first_peaks[i]] < y1_smooth[first_peaks[i - 1]] and y1_smooth[first_peaks[i]] < y1_smooth[
            first_peaks[i + 1]]):
            peaks.append(first_peaks[i])
        elif (y1_smooth[first_peaks[i]] > y1_smooth[first_peaks[i - 1]] and y1_smooth[first_peaks[i]] > y1_smooth[
            first_peaks[i + 1]]):
            peaks.append(first_peaks[i])

    peaks.append(first_peaks[-1])
    # print(files[num*24],phase)
    for i in range(0, len(peaks) - 1):
        peak_max = max(y1_smooth[peaks[i]], y1_smooth[peaks[i + 1]])
        peak_min = min(y1_smooth[peaks[i]], y1_smooth[peaks[i + 1]])

        if y1_smooth[peaks[i]] > y1_smooth[peaks[i + 1]]:
            index_max = peaks[i]
            index_min = peaks[i + 1]
        else:
            index_max = peaks[i + 1]
            index_min = peaks[i]

        if peak_max - peak_min > 50:
            index_max_list.append(index_max)
            index_min_list.append(index_min)

    active_strike_list = []
    for i in range(len(index_max_list)):
        if index_max_list[i] < index_min_list[i]:
            state = 'down'
            active_strike = y1_smooth[index_min_list[i]] - y1_smooth[index_max_list[i]]

        else:
            state = 'up'
            active_strike = y1_smooth[index_max_list[i]] - y1_smooth[index_min_list[i]]
        if active_strike < 0:
            active_strike *= -1
        index1 = min(index_max_list[i], index_min_list[i])
        index2 = max(index_max_list[i], index_min_list[i])
        active_strike_list.append(active_strike)
        # ax.scatter(index_max_list[i], y1_smooth[index_max_list[i]], marker='x', s=50)
        # ax.scatter(index_min_list[i], y1_smooth[index_min_list[i]], marker='o', s=50)
        # ax2.scatter(index_max_list[i], y1[index_max_list[i]], marker='x', s=50)
        # ax2.scatter(index_min_list[i], y1[index_min_list[i]], marker='o', s=50)
        ts_list.append([index1, index2, active_strike, state])
    # plt.show()
    for i in range(len(ts_list) - 1):
        if (ts_list[i][-1] == 'up') & (ts_list[i + 1][-1] == 'down'):
            if abs(ts_list[i][2] - ts_list[i + 1][2]) < 0.25 * max(ts_list[i][2], ts_list[i + 1][2]):
                a = ts_list[i][0]
                b = ts_list[i + 1][1]
                if b - a < 10000:
                    # print(ts_list[i],'\t',ts_list[i+1])
                    frag = y1[a:b]
                    frag = [x - min(frag) for x in frag]
                    if (np.max(frag) - np.min(frag)) > 50:
                        wtr_list.append(frag)


def err_call_back(err):
    print(f'出错啦~ error：{str(err)}')


if __name__ == '__main__':
    file_path = r'E:\data\1.5y data'
    files = glob.glob(os.path.join(file_path, "*.csv"))
    files.sort()
    pbar = tqdm(total=len(files) // 24 * 3, ncols=120)
    pbar.set_description('Rate of Progress')
    update = lambda *args: pbar.update()
    p = Pool(5)
    result_list = Manager().list()
    params_combinations = product(range(1, len(files) // 24 + 1), range(1, 4))
    for param1, param2 in params_combinations:
        p.apply_async(draw_figure, args=(param1, param2, files, result_list), callback=update,
                      error_callback=err_call_back)

    p.close()
    p.join()
    pbar.close()
    wrt_list = list(result_list)
    with open('ts.csv', mode='w', newline='\n') as f:
        writer = csv.writer(f)
        writer.writerows(wrt_list)
    exit()
