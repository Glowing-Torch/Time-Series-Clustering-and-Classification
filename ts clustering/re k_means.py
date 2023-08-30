from multiprocessing import Pool, Manager
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import silhouette_score
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
import pandas as pd
import math
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def find_key_by_value(dct, value):
    return [key for key, val in dct.items() if val == value]
def get_score(ts, i, scores,metric='dtw'):
    # for i in range(3, 10):
    km = TimeSeriesKMeans(n_clusters=i, verbose=False, n_jobs=4, metric=metric)
    y_pred = km.fit_predict(ts)
    if len(ts)==2:
        y_pred=[0,1]
    try:
        score = silhouette_score(ts, y_pred, metric=metric)
    except Exception as e:
        ts_index=tuple(range(len(ts)))
        y_pred=tuple(y_pred)
        dic=dict(zip(ts_index,y_pred))
        values=list(dic.values())
        peaks=[]
        # a=find_key_by_value(dic,0)
        for value in values:
            ind=find_key_by_value(dic,value)[0]
            peak=ts[ind]
            peak=max(peak)
            peaks.append(peak[0])
        if max(peaks)-min(peaks)>150:
            score=0.5
        else:
            score=0
    print(f'Number of clusters:{i} \nsilhouette score:{score}\n')
    scores[i]=score


def select_k(scores):
    # plt.plot(range(10, 20), scores, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Score Line')
    # plt.show()
    k =  max(scores, key=scores.get)
    return k,max(scores.values())


def count_elements(lst):
    counts = {}  # 创建一个空字典用于记录元素计数
    for element in lst:
        if element in counts:
            counts[element] += 1
        else:
            counts[element] = 1
    return counts


def get_labels(k, ts,score):
    km = TimeSeriesKMeans(n_init=5,n_clusters=k, verbose=False, n_jobs=4, metric='dtw')
    y_pred = km.fit_predict(ts)
    result = count_elements(y_pred)
    if score==0.5:
        print(score, result)
        return y_pred
    score = silhouette_score(ts, y_pred, metric='softdtw')
    if score<0.5:
        y_pred=0
    print(score, result)

    # for yi in range(k):
    #     for j in range(len(y_pred)):
    #         if y_pred[j] == yi:
    #             plt.plot(ts[j].ravel(), "k-", alpha=.4)
    #     plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    #     plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
    #              transform=plt.gca().transAxes)
    #     plt.tight_layout()
    #     plt.show()
    return y_pred


def remove_nan(lst):
    for row in lst:
        for i in range(len(row)):
            if math.isnan(row[i]):
                row[i] = 0  # 替换为其他值，或使用 del row[i] 删除
    return lst


def err_call_back(err):
    print(f'出错啦~ error：{str(err)}')


def main(yi,df_list,df):
    dataframe = df.loc[df['label'] == yi, 'value']
    value_array = dataframe.values
    value_array = [eval(value) for value in value_array]
    single_df = pd.DataFrame({'value':value_array})
    length=len(value_array)
    metric='dtw'
    if (length>=2) & (length<=10):
        y = to_time_series_dataset(value_array)
        result_dict = dict()
        if length==2 or length==3:
            end_point=3
            metric='softdtw'
        else:
            end_point=4
        for j in range(2,end_point):
            get_score(y, j, result_dict,metric=metric)
        if max(result_dict.values())>=0.5:
            mse,score = select_k(result_dict)
            labels = get_labels(mse, y,score)
        else:
            labels=0
    else:
        labels=0

    single_df['label'] = labels
    df_list.append(single_df)
    print('Cluster:', yi,'已完成')


if __name__ == '__main__':
    original_index = []
    g_df = pd.read_excel(r'detailed ts with labels.xlsx', engine='openpyxl')
    k = g_df['label'].max()
    df_list = list()
    for yi in range(544,545):
        (main(yi,df_list,g_df))

    df_list=list(df_list)
    for i in range(1,len(df_list)):
        b=df_list[i-1]['label'].max()+1
        df_list[i]['label']+=b
    df = pd.concat(df_list)
    exit()
