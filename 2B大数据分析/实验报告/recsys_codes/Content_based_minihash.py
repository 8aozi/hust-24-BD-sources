from collections import Counter
import math
import numpy as np
import minihash
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix

def tokenize_string(my_string):
    # 正则过滤并小写化，返回合法 token 列表。
    return re.findall(r'[\w\-]+', my_string.lower())

def tokenize(movies):
    # 遍历 movies 的 genres 列，调用 tokenize_string，写入 movies['tokens']。
    tokenlist = []
    for index, row in movies.iterrows():
        tokenlist.append(tokenize_string(row.genres))
    movies['tokens'] = tokenlist
    return movies

def featurize(movies):
    # 构建词表映射 (token -> 索引)
    vocab_set = set()
    for tokens in movies['tokens']:
        for token in tokens:
            vocab_set.add(token)
            
    # 关键点：对特征词进行排序，确保特征矩阵的列顺序在任何环境下绝对一致
    vocab = {token: i for i, token in enumerate(sorted(vocab_set))}
    
    return movies, vocab

def make_predictions(movies, ratings_train, ratings_test, sim_matrix, movies_map):
    # 提前计算好每个用户的平均分，用于回退
    user_mean_dict = ratings_train.groupby('userId')['rating'].mean().to_dict()
    
    # 提前整理好训练集中每个用户的评分列表，加快查找速度
    train_dict = {}
    for _, row in ratings_train.iterrows():
        u, m, r = row['userId'], row['movieId'], row['rating']
        if u not in train_dict:
            train_dict[u] = []
        train_dict[u].append((m, r))

    predictions = []
    for _, row in ratings_test.iterrows():
        u, i = row['userId'], row['movieId']
        # 默认回退值：当前用户均分，如果用户不在训练集，回退至全局平均分 2.5
        pred = user_mean_dict.get(u, 2.5) 

        # 如果待测电影在电影库中，并且用户在训练集有评分记录
        if i in movies_map and u in train_dict:
            idx_i = movies_map[i]
            num = 0.0
            den = 0.0
            for j, r_uj in train_dict[u]:
                if j in movies_map:
                    idx_j = movies_map[j]
                    sim = sim_matrix[idx_i, idx_j]
                    if sim > 0:
                        num += sim * r_uj
                        den += sim
            
            # 只有分母大于 0 时，才使用相似度加权
            if den > 0:
                pred = num / den

        predictions.append(pred)
    return predictions

def make_recommendation(user_id, movies, ratings_train, sim_matrix, movies_map, k):
    """
    对某个用户做前k个推荐
    """
    mlist = list(ratings_train.loc[ratings_train['userId'] == user_id]['movieId'])
    mrlist = list(ratings_train.loc[ratings_train['userId'] == user_id]['rating'])
    movies_list = movies['movieId'].tolist()
    
    # 计算当前用户的平均打分，无打分则回退至 2.5
    user_mean = sum(mrlist) / len(mrlist) if len(mrlist) > 0 else 2.5
    predicted_ratings = {}
    mlist_set = set(mlist)
    
    for m in movies_list:
        if m not in mlist_set:  # 只对未看过的电影进行打分预测
            pred = user_mean
            if m in movies_map:
                idx_i = movies_map[m]
                num = 0.0
                den = 0.0
                # 遍历用户已经打过分的电影
                for j, r_uj in zip(mlist, mrlist):
                    if j in movies_map:
                        idx_j = movies_map[j]
                        sim = sim_matrix[idx_i, idx_j]
                        if sim > 0:
                            num += sim * r_uj
                            den += sim
                if den > 0:
                    pred = num / den
            predicted_ratings[m] = pred

    # 将预测得分从高到低排序，如果预测分相同，则按 movieId 从小到大排序(作为tie-breaker)
    rating_sorted = sorted(predicted_ratings.items(), key=lambda x: (-x[1], x[0]))

    print("recommendation for user " + str(user_id) + ":")
    for i in range(k):
        print("movie_id: " + str(rating_sorted[i][0]) + "; rating: " + str(rating_sorted[i][1]))
    
def sse(predictions, ratings_test):
    # 计算预测误差平方和
    y_true = ratings_test['rating'].to_numpy()
    y_pred = np.array(predictions)
    return np.sum((y_pred - y_true) ** 2)

def get_feature_matrix1(movies, vocab):
    """
    得到n*m的01特征矩阵,n为movies的数量，m为feature的数量
    """
    # 创建一个 n_movies 行 x n_features 列的全 0 矩阵
    matrix = np.zeros((len(movies), len(vocab)))
    
    # 确保直接拿原 DataFrame 的 index 进行精准对应
    for index, row in movies.iterrows():
        for token in row['tokens']:
            if token in vocab:
                matrix[index, vocab[token]] = 1
                
    return matrix

if __name__ == '__main__':
    path = os.path.join('datasets')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    movies_map = {row['movieId']: index for index, row in movies.iterrows()}
    p = int(input())
    print('minhash:')
    sim_matrix = minihash.miniHash(get_feature_matrix1(movies, vocab).T, 5)
    
    ratings_train = pd.read_csv(path + os.path.sep + 'train_set.csv')
    ratings_test = pd.read_csv(path + os.path.sep + 'test_set.csv')
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test, sim_matrix, movies_map)
    print('SSE=%f' % sse(predictions, ratings_test))
    make_recommendation(p, movies, ratings_train, sim_matrix, movies_map, 5)
