import numpy as np
import pandas as pd
import random

def miniHash(arr, h = 20):
    if isinstance(arr, pd.DataFrame):
        arr = arr.to_numpy()  # 将Dataframe类型转化为numpy类型，在取值
    x = arr.shape[0]
    y = arr.shape[1]
    hashmap = [0]*h
    # 固定随机种子，保证不同脚本之间结果可复现、可对齐
    rng = random.Random(42)
    # 生成20个映射函数
    for i in range(h):             
        hashmap[i] = rng.sample(list(range(x)), x)
    
    signarr = np.full((h, y), 9999)
    # TODO: 生成签名矩阵
    # --------------begin---------------------
    for i in range(h):
        # 将 hashmap 的一维列表转为 (x, 1) 形状的列向量，利用 Numpy 广播机制
        hash_vals = np.array(hashmap[i]).reshape(-1, 1)
        # 仅当原矩阵有 1 (即评过分) 时，记录对应的哈希值，否则记为 9999
        masked_hash = np.where(arr == 1, hash_vals, 9999)
        # 求每一列的最小哈希值作为签名
        signarr[i, :] = np.min(masked_hash, axis=0)
    # --------------end---------------------

    simarr = np.zeros((y, y))
    # TODO: 统计相似程度
    # --------------begin---------------------
    for i in range(y):
        # Numpy 广播匹配当前列与其他所有列相等的哈希数量，求均值即为近似 Jaccard 相似度
        simarr[i, :] = np.sum(signarr == signarr[:, i:i+1], axis=0) / h
    # --------------end---------------------
    
    return simarr