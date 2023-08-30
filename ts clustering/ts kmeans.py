import numpy as np
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from tslearn.clustering import silhouette_score
from tslearn.utils import to_time_series_dataset
from multiprocessing import Pool, Manager
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.spatial.distance import euclidean

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def calculate_distance_matrix(time_series):
    n = len(time_series)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            distance, _ = fastdtw(time_series[i], time_series[j])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return distance_matrix


def uniform_sampling(lst, m):
    return [row[::m] for row in lst if max(row)>50]


class FastDTWKMeans(TimeSeriesKMeans):
    def metric(self, X, Y):
        # 调用FastDTW计算距离
        distance, _ = fastdtw(X, Y, dist=euclidean)
        return distance


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
    km = TimeSeriesKMeans(n_clusters=i, verbose=False, n_jobs=4, metric='dtw')
    y_pred = km.fit_predict(ts)
    score = silhouette_score(ts, y_pred, metric='dtw')
    print(f'Number of clusters:{i} \nsilhouette score:{score}\n')
    scores[i]=score


def select_k(scores):
    # plt.plot(range(3, 9), scores, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Score Line')
    # plt.show()
    k = max(scores,key=scores.get)
    return k


def get_labels(k, ts):
    km = TimeSeriesKMeans(n_clusters=k, verbose=False, n_jobs=4, metric='dtw')
    y_pred = km.fit_predict(ts)
    score = silhouette_score(ts, y_pred, metric='dtw')
    result = count_elements(y_pred)
    print(score, result)
    for yi in range(k):
        for j in range(len(y_pred)):
            if y_pred[j] == yi:
                plt.plot(ts[j].ravel(), "k-", alpha=.4)
        plt.plot(km.cluster_centers_[yi].ravel(), "r-")
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.show()
    return y_pred


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
    mse = select_k(result_dict)
    labels = get_labels(mse, y)
    single_df['label'] = labels + benchmark
    benchmark = single_df['label'].max() + 1
    df_list.append(single_df)


if __name__ == '__main__':
    file_name = r'ts.csv'
    data_array = np.load('my_array.npy',allow_pickle=True)
    # data_array = get_ts(file_name)
    interval = 100
    num = len(data_array) // interval
    df_list = []
    i = 0
    while True:
        main(i, data_array, interval)
        i += 1
        if i > num:
            break
    df = pd.concat(df_list)
    # df.to_excel('normalized_ts with labels.xlsx', engine='openpyxl', encoding='utf')
    plt.close()
    exit()
