import numpy as np
import math
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from tslearn.clustering import silhouette_score
from tslearn.utils import to_time_series_dataset
from multiprocessing import Pool, Manager
from tslearn.clustering import KShape,TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.spatial.distance import euclidean

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)



def uniform_sampling(lst, m):
    return [row[::m] for row in lst if max(row)>50]



def remove_nan(lst):
    for row in lst:
        for i in range(len(row)):
            if math.isnan(row[i]):
                row[i] = 0  # 替换为其他值，或使用 del row[i] 删除
    return lst


def count_elements(lst):
    counts = {}  # 创建一个空字典用于记录元素计数
    for element in lst:
        if element in counts:
            counts[element] += 1
        else:
            counts[element] = 1
    return counts


def get_ts(file):
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = [row for row in reader]

    data_array = []
    for row in data:
        data_list = []
        for d in row:
            d = '%.1f' % eval(d)
            data_list.append(eval(d))
        data_array.append(data_list)
    np.save('my_array.npy', data_array)
    return data_array


def get_score(ts, i, scores):
    # for i in range(3, 10):
    km = TimeSeriesKMeans(n_clusters=i, verbose=False,metric='dtw')
    y_pred = km.fit_predict(ts)
    score = silhouette_score(ts, y_pred, metric='dtw')
    print(f'Number of clusters:{i} \nsilhouette score:{score}\n')
    scores[i]=score



def select_k(scores):
    global end_time
    end_time=time.time()
    plt.plot(range(3, 9), scores.values(), marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score Line')
    plt.show()
    k = max(scores,key=scores.get)
    return k,max(scores.values())




def main(i, data_array, interval):
    global benchmark
    print(i)
    if i == 0:
        benchmark = 0
    if i < num:
        frag_data_array = data_array[i * interval:(i + 1) * interval]
    else:
        frag_data_array = data_array[num * interval:]
    frag_data_array = uniform_sampling(frag_data_array, 20)
    single_df = pd.DataFrame({'value': frag_data_array})
    x = to_time_series_dataset(frag_data_array)
    y = remove_nan(x)
    y= TimeSeriesScalerMeanVariance().fit_transform(y)
    p = Pool(6)
    result_dict = Manager().dict()
    for j in range(3, 9):
        p.apply_async(get_score, args=(y, j, result_dict))
    p.close()
    p.join()
    mse,best_score = select_k(result_dict)
    print(mse,best_score)
    # labels = get_labels(mse, y)
    # single_df['label'] = labels + benchmark
    # benchmark = single_df['label'].max() + 1
    # df_list.append(single_df)


if __name__ == '__main__':
    start_time=time.time()
    file_name = r'ts.csv'
    data_array = np.load('my_array.npy',allow_pickle=True)
    interval = 100
    num = len(data_array) // interval
    df_list = []
    i = 0
    while i<1:
        main(i, data_array, interval)
        i += 1
        if i > num:
            break
    print(end_time-start_time)
    plt.close()
    exit()
