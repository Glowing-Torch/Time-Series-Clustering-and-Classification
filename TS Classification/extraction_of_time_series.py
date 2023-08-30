import glob
import os
from itertools import product
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from multiprocessing import Pool, Manager

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ts_extraction():
    def __init__(self, date):
        self.date = date
        self.dataset = []
        self.ts_list=[]

    def choose_file(self):
        file_path = r'E:\data\1.5y data'
        files = glob.glob(os.path.join(file_path, self.date))
        files.sort()
        self.files = files

    def read_files(self):
        for f in self.files:
            df = pd.read_csv(f, delimiter=';')
            self.dataset.append(df)
        df = pd.concat(self.dataset)
        ph = f'total_power_active:W'
        y1 = np.array(df[ph])
        x = np.arange(1, df.shape[0] + 1)
        self.x = x
        self.y1 = y1

    def find_peaks(self):
        y1 = self.y1
        for i in range(1, len(y1) - 1):
            if (y1[i] - y1[i - 1] > 20) and (y1[i] - y1[i + 1] > 20):
                y1[i] = y1[i - 1]
        y1_smooth = savgol_filter(y1, 59, 1)
        first_peaks = []
        for i in range(1, len(y1_smooth) - 1):
            if (y1_smooth[i] < y1_smooth[i - 1]) and (y1_smooth[i] < y1_smooth[i + 1]):
                first_peaks.append(i)
            elif (y1_smooth[i] > y1_smooth[i - 1]) and (y1_smooth[i] > y1_smooth[i + 1]):
                first_peaks.append(i)
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
        self.peaks=peaks
        self.y1_smooth=y1_smooth
    def cal_diff(self):
        peaks=self.peaks
        y1_smooth=self.y1_smooth
        index_max_list = []
        index_min_list = []
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
            self.ts_list.append([index1, index2, active_strike, state])
    def ts_draw(self):
        draw_part=[]
        ts_list=self.ts_list
        for i in range(len(ts_list) - 1):
            if (ts_list[i][-1] == 'up') & (ts_list[i + 1][-1] == 'down'):
                if abs(ts_list[i][2] - ts_list[i + 1][2]) < 0.3 * max(ts_list[i][2], ts_list[i + 1][2]):
                    a = ts_list[i][0]
                    b = ts_list[i + 1][1]
                    if b - a < 10000:
                        frag = self.y1[a:b]
                        frag2 = [x - min(frag) for x in frag]
                        if (np.max(frag2) - np.min(frag2)) > 50:
                            draw_part.append([range(a, b), frag])
        self.draw_part=draw_part
    def draw_pic(self):
        plt.plot(self.x,self.y1)
        plt.xlabel('Sekunde/s')
        plt.ylabel('Wirkleistung/W')
        return max(self.y1)-0.2
    def run(self):
        self.choose_file()
        self.read_files()
        self.find_peaks()
        self.cal_diff()
        self.ts_draw()
        return self.draw_part


if __name__=='__main__':
    date = r'2023-06-28'
    date += '*'
    a = ts_extraction(date)
    a.run()


