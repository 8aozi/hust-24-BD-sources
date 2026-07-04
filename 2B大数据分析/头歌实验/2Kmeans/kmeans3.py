# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import os

# 自定义准确率计算函数
def accuracy_score(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)


# global value
SetName = '葡萄酒识别'
wineFeat = ['酒精', '苹果酸', '灰', '灰分的灰度', '镁', '总酚', '黄酮', 
            '非类黄酮酚', '原花青素', '颜色强度', '色调', '稀释葡萄酒的OD280 / OD315',
            '脯氨酸']
colors = ['b', 'g', 'r', '#e24fff', '#524C90', '#845868', 
          'k', 'c', 'm', 'y', ]
k = 3       # 质心数量
xlabel = 4  # 可视化的特征维度
ylabel = 5


def regularit(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        d = pd.to_numeric(d)
        MAX = d.max()
        MIN = d.min()
        newDataFrame[c] = ((d - MIN + 1e-7) / (MAX - MIN)).tolist()
    return newDataFrame


def loadDataSet(FilePath, norm=False):
    df = pd.read_csv(FilePath, sep=',', header=None, dtype=str, index_col=0,
                     na_filter=False)
    if norm:
        df = regularit(df)
        df.to_csv('归一化数据.csv', index=True, header=False)
    return np.array(df).astype(float), np.array(df.index)


def plot(data, labels, cents, k, sse, acc):
    """
    功能：可视化K-means聚类结果
    输入：
        data: 样本数据 (n_samples, n_features)
        labels: 聚类标签 (n_samples,)
        cents: 质心坐标 (k, n_features)
        k: 质心数量
        sse: 误差平方和
        acc: 准确率
    
        生成聚类结果图片
    """
    
    # ========== 你需要完成 ==========
    # 提示：
    # 1. 设置图表标题：plt.title("SSE={:.3f}  Acc={:.3f}".format(sse, acc))
    # 2. 设置x轴和y轴标签：plt.xlabel(wineFeat[xlabel]), plt.ylabel(wineFeat[ylabel])
    # 3. 设置中文字体：plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 4. 对每个簇（i从0到k-1）：
    #    - 找出属于当前簇的样本索引：index = np.nonzero(labels == i)[0]
    #    - 绘制样本点：plt.scatter(x=data[index, xlabel], y=data[index, ylabel],
    #                color=colors[i], linewidths=0.1)
    #    - 绘制质心点：plt.scatter(cents[i, xlabel], cents[i, ylabel], 
    #                marker='x', color=colors[i], linewidths=10)
    # 5. 图片生成
    #    outFile = str(k) + '类' + SetName + '-' + wineFeat[xlabel] \
    #              + '-' + wineFeat[ylabel] + ".png"
    #    plt.savefig(outFile)
    #    不需要将图片保存、输出
    #plt.savefig(outFile)
    #plt.show()
    #如果需要进行本地测试的话，上面两行代码可以将图片保存、输出。但是头歌测试平台的结果并不需要将图片输出。
    # ========== begin ==========
def plot(data, labels, cents, k, sse, acc):
    plt.title("SSE={:.3f}  Acc={:.3f}".format(sse, acc))
    plt.xlabel(wineFeat[xlabel])
    plt.ylabel(wineFeat[ylabel])
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    for i in range(k):
        index = np.nonzero(labels == i)[0]
        plt.scatter(x=data[index, xlabel], y=data[index, ylabel],
                    color=colors[i], linewidths=0.1)
        plt.scatter(cents[i, xlabel], cents[i, ylabel],
                    marker='x', color=colors[i], linewidths=10)


    # ========== end ============

def calEDist(arrA, arrB):
    """功能：欧拉距离距离计算，输入：两个一维数组"""
    # 修改点：np.math.sqrt -> math.sqrt
    return math.sqrt(sum(np.power(arrA - arrB, 2)))


def randCent(data_X, k, rand_state):
    n = data_X.shape[1]        
    centroids = np.empty((k, n))
    random.seed(rand_state)
    index = random.sample(list(range(len(data_X))), k)
    for i in range(len(index)):
        centroids[i, :] = data_X[index[i], :]
    return centroids


def k_means(data_X, k, rand_state=20214234, max_iter=500, initCent='random'):
    m = data_X.shape[0]
    clusterAssment = np.zeros((m, 2)) 

    if initCent == 'random':
        centroids = randCent(data_X, k, int(rand_state))

    clusterChanged = True
    for _ in range(max_iter):
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                arrA = centroids[j, :]
                arrB = data_X[i, :]
                dist = calEDist(arrA, arrB)
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2        
        if not clusterChanged:
            break
        
        for i in range(k):
            index_all = clusterAssment[:, 0]
            value = np.nonzero(index_all == i)
            ptsInClust = data_X[value[0]]
            centroids[i, :] = np.mean(ptsInClust, axis=0)
    labels = clusterAssment[:, 0]
    sse = sum(clusterAssment[:, 1])
    return labels, centroids, sse


def bestSeed(data, true_lbl, flag, iter=100, initType='random'):
    accList = []
    sseList = []
    for i in range(flag):
        labels, _, sse = k_means(data, k, rand_state=i, max_iter=iter, initCent=initType)
        acc = accuracy_score(true_lbl, labels+1)
        accList.append(acc)
        sseList.append(sse)
    max_idx = np.argmax(accList)
    min_idx = np.argmin(sseList)
    return min_idx, max_idx


if __name__ == "__main__":
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    #print(f"当前工作目录: {os.getcwd()}")
    #print(f"当前目录下的文件: {os.listdir('.')}")
    
    # 使用已存在的归一化数据
    File = "/data/bigfiles/归一化数据.csv"
    
    if not os.path.exists(File):
        print(f"错误：找不到文件 '{File}'")
        exit(1)
    
    #print(f"\n正在加载数据文件: {File}")
    data, true_lbl = loadDataSet(File, norm=False)
    
    # 寻找最佳种子
    #print("\n正在寻找最佳初始种子...")
    sse_idx, acc_idx = bestSeed(data, true_lbl, 5)
    
    # 运行K-means
    #print("\n正在运行K-means算法...")
    labels, cents, sse = k_means(data, k, rand_state=acc_idx, max_iter=100, initCent='random')
    
    # 计算准确率
    max_acc = 0
    best_mapping = 0
    #print("\n准确率计算（尝试不同的标签映射）:")
    for i in range(3): 
        acc = accuracy_score(true_lbl, (labels + i) % 3 + 1)
        #print(f"  映射 {i}: (labels+{i})%3+1 -> 准确率 = {acc:.4f}")
        if acc > max_acc:
            max_acc = acc
            best_mapping = i
    
    # 绘制结果
    
    try:
        plot(data, labels, cents, k, sse, max_acc)
        #print(f"\n图片已保存")
        print("SSE={:.3f}  Acc={:.3f}".format(sse, acc))
    except Exception as e:
        print(f"\n绘图失败: {e}")
    