import numpy as np
from scipy.cluster import hierarchy
from tslearn.metrics import cdist_dtw
import matplotlib.pyplot as plt
from tslearn.clustering import silhouette_score
from tslearn.utils import to_time_series_dataset
import pandas as pd

def ts_feature():
    global original_index
    x_new_data=[]
    s_new_data = []
    s_original_index=[]
    x_original_index=[]
    xx_new_data = []
    xx_original_index=[]
    for number in num_list:
        ts_df = df.loc[df['label'] == number, 'value']
        ts = ts_df.iloc[0]
        ts = eval(ts)
        if max(ts)<50:
            continue
        elif max(ts) < 500:
            s_new_data.append(ts)
            ts_index = ts_df.axes[0][0]
            s_original_index.append(ts_index)
        elif max(ts)<2000:
            x_new_data.append(ts)
            ts_index = ts_df.axes[0][0]
            x_original_index.append(ts_index)
        else:
            xx_new_data.append(ts)
            ts_index = ts_df.axes[0][0]
            xx_original_index.append(ts_index)
    x_new_data = to_time_series_dataset(x_new_data)
    s_new_data = to_time_series_dataset(s_new_data)
    xx_new_data=to_time_series_dataset(xx_new_data)
    return s_new_data, s_original_index,x_new_data,x_original_index,xx_new_data,xx_original_index


def plt_tree(iter_res):
    plt.figure(dpi=300, figsize=(10, 6))
    plt.axhline(y=5, color='r', linestyle='--')
    dn = hierarchy.dendrogram(iter_res)
    plt.show()


def cal_dis(ts_array,i):
    dist_matrix = cdist_dtw(ts_array, verbose=True)
    if i==1:
        np.save('under 500W distance matrix.npy', dist_matrix)
    elif i==2:
        np.save('500-2000W distance matrix.npy', dist_matrix)
    elif i==3:
        np.save('over 2000W distance matrix.npy', dist_matrix)
    return dist_matrix


def cut_tree(iter_res, threshold, ts_array):
    labels = hierarchy.cut_tree(iter_res, height=threshold)
    labels = [num for sublist in labels for num in sublist]

    k = max(labels)
    print(k)
    # for yi in range(k+1):
    #     for j in range(len(labels)):
    #         if labels[j] == yi:
    #             plt.plot(ts_array[j].ravel(), "k-", alpha=.4)
    #     plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
    #              transform=plt.gca().transAxes)
    #     plt.tight_layout()
    #     plt.show()
    return labels


def find_max(arr):
    arr=eval(arr)
    return max(arr)


if __name__ == '__main__':
    original_index = []
    df = pd.read_excel(r'detailed ts with labels.xlsx', engine='openpyxl')
    # va=df['value']
    df=df[df['value'].apply(find_max)>50]
    num_list = df['label'].unique()
    s_data_array, s_index_list,x_data_array,x_index_list,xx_data_array,xx_index_list = ts_feature()
    index_list=s_index_list+x_index_list+xx_index_list
    ori_labels = np.array(df.loc[index_list, 'label'])

    # dist_matrix_1=cal_dis(s_data_array,1)
    dist_matrix_1 = np.load('under 500W distance matrix.npy', allow_pickle=True)
    Z1 = hierarchy.linkage(dist_matrix_1, method='ward')  # 使用Ward方法进行层次聚类
    plt_tree(Z1)
    labels_1 = cut_tree(Z1, 12000, s_data_array)
    benchmark_1=max(labels_1)+1

    # dist_matrix_2=cal_dis(x_data_array,2)
    dist_matrix_2 = np.load('500-2000W distance matrix.npy', allow_pickle=True)
    Z2 = hierarchy.linkage(dist_matrix_2, method='ward')  # 使用Ward方法进行层次聚类
    plt_tree(Z2)
    labels_2= cut_tree(Z2, 48000, x_data_array)
    labels_2=[l+benchmark_1 for l in labels_2]
    benchmark_2=max(labels_2)+1

    # dist_matrix_3 = cal_dis(xx_data_array, 3)
    dist_matrix_3 = np.load('over 2000W distance matrix.npy', allow_pickle=True)
    Z3 = hierarchy.linkage(dist_matrix_3, method='ward')  # 使用Ward方法进行层次聚类
    plt_tree(Z3)
    labels_3 = cut_tree(Z3, 26000, xx_data_array)
    labels_3=[l+benchmark_2 for l in labels_3]
    # print(mapped_dict)

    labels=labels_1+labels_2+labels_3
    k=max(labels_3)
    mapped_dict=dict(zip(ori_labels,labels))

    df['new_label']=df['label'].map(mapped_dict)
    print(k)
    for yi in range(k+1):
        dataframe=df.loc[df['new_label']==yi,'value']
        value_array=np.array(dataframe)
        value_array=[eval(value) for value in value_array]
        value_array=to_time_series_dataset(value_array)
        for value in value_array:
            plt.plot(value.ravel(), "k-", alpha=.4)
        plt.text(0.55, 0.85, 'Cluster %d' % (yi + 1),
                 transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(rf'C:\Users\86137\Desktop\Bachelorarbeit\pictures\Cluster {yi+1}',dpi=300)
        plt.show()
    df.to_excel('final time series.xlsx',engine='openpyxl',encoding='utf8')
    exit()
