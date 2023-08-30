import numpy as np
from collections import Counter
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class edge_extraction():
    def __init__(self,file):
        self.dataset=[]
        self.test_part=[]
        self.file=file
        self.y1=None
        self.y2=None
        self.y1_smooth=None
        self.y2_smooth=None
        self.index_max_list=[]
        self.index_min_list=[]

    def read_file(self):
        data_list = []
        total_power_active = []
        total_power_reactive = []
        with open(self.file,encoding='utf8') as f1:
            lines = f1.readlines()
            for line in lines[1:]:
                data_list.append(line)

        for data in data_list:
            data = data.split(';')
            total_power_active.append(eval(data[25]))
            total_power_reactive.append(eval(data[27]))
        self.y1,self.y2=total_power_active,total_power_reactive

    def smooth_process(self):
        list1=self.y1
        list2=self.y2
        for i in range(1,len(list1)-1):
            if (list1[i] - list1[i - 1] > 20) and (list1[i] - list1[i + 1] > 20):
                list1[i] = list1[i - 1]
            if (list2[i] - list2[i - 1] > 20) and (list2[i] - list2[i + 1] > 20):
                list2[i] = list2[i - 1]
        y1_smooth = savgol_filter(list1, 59, 1)
        y2_smooth=savgol_filter(list2,59,1)
        self.y1_smooth,self.y2_smooth= y1_smooth,y2_smooth

    def find_peaks(self):
        y1_smooth=self.y1_smooth
        y1=self.y1
        first_peaks=[]
        index_max_list = []
        index_min_list = []
        for i in range(1, len(y1_smooth) - 1):
            if (y1_smooth[i] <y1_smooth[i - 1] and y1_smooth[i] < y1_smooth[i + 1]):
                first_peaks.append(i)
            elif (y1_smooth[i] > y1_smooth[i - 1] and y1_smooth[i] > y1_smooth[i + 1]):
                first_peaks.append(i)
        for i in range(3):
            disorder_list = []
            for i in range(1, len(first_peaks) - 1):
                delta1 = y1_smooth[first_peaks[i]] - y1_smooth[first_peaks[i - 1]]
                delta2 = y1_smooth[first_peaks[i]] - y1_smooth[first_peaks[i + 1]]
                if (delta1 < 0 and delta2 < 0):
                    if (delta1 > -10 or delta2 > -10):
                        disorder_list.append(i)
            if disorder_list == []:
                break
            else:
                i = 1
                for disorder in disorder_list:
                    i -= 1
                    del first_peaks[disorder + i]
        peaks = list()
        peaks.append(first_peaks[0])
        values_list=[]
        for peak in first_peaks:
            values_list.append(int(y1_smooth[peak]))
        c = dict(Counter(values_list))
        for i in range(1, len(first_peaks) - 1):
            if c[int(y1_smooth[first_peaks[i]])] >= 5:
                peaks.append(first_peaks[i])
            elif (y1_smooth[first_peaks[i]] < y1_smooth[first_peaks[i - 1]] and y1_smooth[first_peaks[i]] < y1_smooth[
                first_peaks[i + 1]]):
                peaks.append(first_peaks[i])
            elif (y1_smooth[first_peaks[i]] > y1_smooth[first_peaks[i - 1]] and y1_smooth[first_peaks[i]] > y1_smooth[
                first_peaks[i + 1]]):
                peaks.append(first_peaks[i])
        peaks.append(first_peaks[-1])
        for i in range(0,len(peaks)-1 ):
            real_peak_max = max(y1[peaks[i]:peaks[i + 1]])
            real_peak_min = min(y1[peaks[i]:peaks[i + 1]])
            peak_max = max(y1_smooth[peaks[i]],y1_smooth[peaks[i +1]])
            peak_min = min(y1_smooth[peaks[i]],y1_smooth[peaks[i + 1]])
            index_max = y1_smooth.tolist().index(peak_max)
            index_min = y1_smooth.tolist().index(peak_min)
            if (peak_max - peak_min > 30) and (real_peak_max - real_peak_min > 30):
                if peak_max < 1000:
                    index_max_list.append(index_max)
                    index_min_list.append(index_min)

            if (peak_max - peak_min > 30) and (peak_max > 1000):
                if abs((peak_max - peak_min - real_peak_max + real_peak_min) / (real_peak_max - real_peak_min)) < 0.15:
                    index_max_list.append(index_max)
                    index_min_list.append(index_min)
        self.index_max_list,self.index_min_list=index_max_list, index_min_list

    def diff_calculation(self):
        y1=self.y1
        y2=self.y2
        y1_smooth=self.y1_smooth
        y2_smooth=self.y2_smooth
        index_max_list, index_min_list=self.index_max_list,self.index_min_list
        active_strike_list=[]
        reactive_strike_list=[]
        volatile = 0
        for i in range(len(index_max_list)):
            if (i == 0) & (len(index_max_list) == 1):
                bw_index = 0
                fw_index = -1
            elif (i == 0) & (len(index_max_list) > 1):
                bw_index = 0
                fw_index = min(index_max_list[i + 1], index_min_list[i + 1])
            elif (i > 0) & (i < len(index_max_list) - 1):
                bw_index = max(index_max_list[i - 1], index_min_list[i - 1])
                fw_index = min(index_max_list[i + 1], index_min_list[i + 1])
            elif (i == len(index_max_list) - 1) & (i > 0):
                bw_index = max(index_max_list[i - 1], index_min_list[i - 1])
                fw_index = -1
            y1_part = y1[bw_index:fw_index]
            y2_part = y2[bw_index:fw_index]
            if index_max_list[i] < index_min_list[i]:
                draw_index=range(index_max_list[i],index_min_list[i])
                draw_value=y1[index_max_list[i]:index_min_list[i]]
                reactive_strike = y2_smooth[index_min_list[i]] - y2_smooth[index_max_list[i]]
                active_strike = y1_smooth[index_min_list[i]] - y1_smooth[index_max_list[i]]
            else:
                draw_index=range(index_min_list[i],index_max_list[i])
                draw_value=y1[index_min_list[i]:index_max_list[i]]
                reactive_strike = y2_smooth[index_max_list[i]] - y2_smooth[index_min_list[i]]
                active_strike = y1_smooth[index_max_list[i]] - y1_smooth[index_min_list[i]]
            if active_strike<0:
                active_strike *= -1
                reactive_strike *= -1
            range2 = np.max(y2_part) - np.min(y2_part)
            range1 = np.max(y1_part) - np.min(y1_part)
            if (range2 - abs(reactive_strike) > 50) or (range1 > 3 * abs(active_strike)):
                volatile = 1
            active_strike_list.append(active_strike)
            reactive_strike_list.append(reactive_strike)
            self.dataset.append([active_strike,reactive_strike,volatile])
            self.test_part.append([draw_index,draw_value])
        if (len(active_strike_list) == 0) & (abs(y1_smooth[-1] - y1_smooth[0]) > 30):
            if y1_smooth[-1] - y1_smooth[0] < 0:
                active_strike_list.append(y1_smooth[0] - y1_smooth[-1])
                reactive_strike_list.append(y2_smooth[0] - y2_smooth[-1])

            else:
                active_strike_list.append(y1_smooth[-1] - y1_smooth[0])
                reactive_strike_list.append(y2_smooth[-1] - y2_smooth[0])
            self.dataset.append([active_strike_list[0], reactive_strike_list[0],volatile])
    def plot_chart(self):
        plt.plot(range(len(self.y1)),self.y1)
        plt.xlabel('Sekunde/s')
        plt.ylabel('Wirkleistung/W')
        return max(self.y1)-0.2
        # plt.show()
    def run(self):
        self.read_file()
        self.smooth_process()
        self.find_peaks()
        self.diff_calculation()
        return self.dataset,self.test_part


if __name__ == '__main__':
    file_name = r'E:\data\1.5y data\2023-06-29_00-24-59.csv'
    data = edge_extraction(file_name)
    print(data.run())
