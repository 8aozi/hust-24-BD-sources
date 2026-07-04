# -*- coding: utf-8 -*-

import numpy as np
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


def calEDist(arrA, arrB):
    """功能：欧拉距离距离计算，输入：两个一维数组"""
    return math.sqrt(sum(np.power(arrA - arrB, 2)))


def randCent(data_X, k, rand_state):
    """
    功能：随机选取k个质心
    输入：
        data_X: 数据集，形状为 (m, n)，m个样本，n个特征
        k: 质心数量
        rand_state: 随机种子，用于结果可复现
    输出：
        centroids: 返回一个 k*n 的质心矩阵
    """
    n = data_X.shape[1]  # 获取特征的维数
    
    # ========== 你需要完成==========
    # ========== begin==========

    m = data_X.shape[0]

# 1. 创建一个空的质心矩阵：centroids = np.empty((k, n))
    centroids = np.empty((k, n))

 # 2. 设置随机种子：random.seed(rand_state)
    random.seed(rand_state)

    # 3. 从 data_X 中随机选择 k 个不同的样本作为初始质心
    #    - 使用 random.sample() 从样本索引中随机选择 k 个
    #    - 将选中的样本赋值给 centroids
    #    - 返回centroids 
    idx=random.sample(range(m),k)

    centroids = data_X[idx]

    return centroids
    # ========== end ===========
    #raise NotImplementedError("randCent 函数尚未实现，请完成代码")
    #请在完成代码后将上面这行注释掉，避免无法通过测试集。
    #return centroids


def k_means(data_X, k, rand_state=20214234, max_iter=500, initCent='random'):
    m = data_X.shape[0]
    clusterAssment = np.zeros((m, 2)) 

    if initCent == 'random':
        centroids = randCent(data_X, k, int(rand_state))
    # ========== 你需要完成以下代码 =========
   
    # ========== begin ==========

    clusterChanged = True 
    # 1. 使用 for _ in range(max_iter): 进行迭代
    for _ in range(max_iter):
         # 2. 在每个迭代中，先假设簇分配未改变 clusterChanged = False
        clusterChanged = False
         # 3. 对每个样本点 (i从0到m-1)：
        #    - 计算到每个质心的欧氏距离（调用 calEDist 函数）
        #    - 找到最近的质心 (minIndex) 和最小距离 (minDist)
        #    - 如果当前分配的簇发生变化，设置 clusterChanged = True
        #    - 更新 clusterAssment[i] = [minIndex, minDist**2]
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
         # 4. 如果 clusterChanged == False，则提前结束循环 (break)    
        if not clusterChanged:
            break

         # 5. 更新质心：对每个簇，计算簇内所有样本的均值作为新质心
        #    - 使用 np.nonzero 找出属于当前簇的样本索引
        #    - 计算这些样本的均值 np.mean(axis=0)
        for i in range(k):
            index_all = clusterAssment[:, 0]
            value = np.nonzero(index_all == i)
            ptsInClust = data_X[value[0]]
            centroids[i, :] = np.mean(ptsInClust, axis=0)
    labels = clusterAssment[:, 0]
    sse = sum(clusterAssment[:, 1])
    return labels, centroids, sse

   # ========== end ============

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
    
    # 使用已存在的归一化数据
    File = "/data/bigfiles/归一化数据.csv"
    
    if not os.path.exists(File):
        print(f"错误：找不到文件 '{File}'")
        exit(1)
    
    data, true_lbl = loadDataSet(File, norm=False)
    print(f"数据加载成功!")
    print(f"  - 样本数量: {data.shape[0]}")
    print(f"  - 特征维度: {data.shape[1]}")
    print(f"  - 标签值: {np.unique(true_lbl)}")
    
    # 寻找最佳种子
    print("\n正在寻找最佳初始种子...")
    sse_idx, acc_idx = bestSeed(data, true_lbl, 5)
    print(f"最佳种子（按准确率）: {acc_idx}")

    
    # 运行K-means
    print("\n正在运行K-means算法...")
    labels, cents, sse = k_means(data, k, rand_state=acc_idx, max_iter=100, initCent='random')
    
    # 计算准确率
    max_acc = 0
    best_mapping = 0
    print("\n准确率计算（尝试不同的标签映射）:")
    for i in range(3): 
        acc = accuracy_score(true_lbl, (labels + i) % 3 + 1)
        print(f"  映射 {i}: (labels+{i})%3+1 -> 准确率 = {acc:.4f}")
        if acc > max_acc:
            max_acc = acc
            best_mapping = i
    
    # 输出最终结果
    print("\n" + "="*50)
    print("最终结果:")
    print(f"  最佳准确率: {max_acc:.4f}")
    print(f"  误差平方和(SSE): {sse:.4f}")
    print(f"  最佳标签映射: +{best_mapping}")
    print("="*50)
