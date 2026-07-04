import pandas as pd
import numpy as np

# 假设原代码中包含 minihash, 这里省略不需要的部分
movies = pd.read_csv('datasets/movies.csv', index_col=0)

rating_dic = {}
rating_dic_01 = {}
rating_train = open('datasets/train_set.csv', 'r', encoding='UTF-8')
rating_test = pd.read_csv('datasets/test_set.csv')

# 用户-电影效用矩阵
for item in rating_train.readlines()[1:]:
    item = item.strip().split(',')
    if item[0] not in rating_dic.keys():
        rating_dic[item[0]] = {item[1]: item[2]}
    else:
        rating_dic[item[0]][item[1]] = item[2]
        
    if item[0] not in rating_dic_01.keys():
        if float(item[2]) < 3.0:
            rating_dic_01[item[0]] = {item[1]: 0}
        else:
            rating_dic_01[item[0]] = {item[1]: 1}
    else:
        if float(item[2]) < 3.0:
            rating_dic_01[item[0]][item[1]] = 0
        else:
            rating_dic_01[item[0]][item[1]] = 1

uti = pd.DataFrame(rating_dic, dtype='float').T.fillna(0)
user_count = uti.shape[0]
movie_count = uti.shape[1]
uti_jaccard = pd.DataFrame(rating_dic_01).T.fillna(0).astype(int)


def pearson_sim():
    # 1. 创建 user_count x user_count 的相似度矩阵 similar
    # 2. 计算任意两个用户评分向量的 Pearson 相关系数
    # np.corrcoef 默认计算每行之间的皮尔逊相关系数，恰好对应 User-User
    similar = np.corrcoef(uti.values)
    # 处理可能由于除零导致出现的 NaN 异常值（如果某个用户所有电影评分一致，方差为0）
    similar = np.nan_to_num(similar)
    return similar


def recommend(userID, sim_matrix, k_sim_user=10, topn_rec_movies=5):
    userID = str(userID)
    if userID not in uti.index:
        return []
    
    users_list = list(uti.index)
    movie_cols = list(uti.columns)
    user_idx = users_list.index(userID)
    
    # 1. 找到目标用户最相近的 k 个用户
    user_sims = sim_matrix[user_idx].copy()
    user_sims[user_idx] = -np.inf # 排除自身
    top_k_indices = np.argsort(user_sims)[::-1][:k_sim_user]
    
    # 2. 找出目标用户未评分电影
    user_ratings = uti.values[user_idx]
    unrated_movie_indices = np.where(user_ratings == 0)[0]
    
    recommendations = []
    
    # 3. 按相似度加权平均预测评分
    for m_idx in unrated_movie_indices:
        numerator = 0.0
        denominator = 0.0
        for v_idx in top_k_indices:
            rating_vm = uti.values[v_idx, m_idx]
            if rating_vm > 0: # 相似用户给该电影打过分
                sim_uv = sim_matrix[user_idx, v_idx]
                numerator += sim_uv * rating_vm
                denominator += sim_uv
                
        if denominator == 0:
            continue
            
        pred_rating = numerator / denominator
        pred_rating = round(pred_rating, 4)
        
        m_id = int(movie_cols[m_idx])
        recommendations.append((m_id, pred_rating))
        
    # 按预测评分从高到低排序，如果评分相同则按照movieId从小到大排序
    recommendations.sort(key=lambda x: (-x[1], x[0]))
    
    # 4. 返回前 topn_rec_movies 个推荐
    top_n = recommendations[:topn_rec_movies]
    
    res = []
    for m_id, p_rating in top_n:
        row = movies.loc[m_id]
        title = row['title']
        genres = row['genres']
        res.append((m_id, title, genres, p_rating))
        
    return res


def prediction_test_set(sim_matrix, k_sim_user):
    predictions = []
    
    user_id2idx = {u: i for i, u in enumerate(uti.index)}
    movie_id2idx = {m: i for i, m in enumerate(uti.columns)}
    
    # 为了加速测试集预测，缓存用户的 top-k 索引
    top_k_cache = {}
    
    # 1. 遍历 rating_test 中每个 userId-movieId
    for index, row in rating_test.iterrows():
        u_id = str(int(row['userId']))
        m_id = str(int(row['movieId']))
        
        # 处理可能遇到训练集中没有的用户或电影的极端情况
        if u_id not in user_id2idx or m_id not in movie_id2idx:
            predictions.append(2.5)
            continue
            
        u_idx = user_id2idx[u_id]
        m_idx = movie_id2idx[m_id]
        
        if u_idx not in top_k_cache:
            sims = sim_matrix[u_idx].copy()
            sims[u_idx] = -np.inf
            top_k_cache[u_idx] = np.argsort(sims)[::-1][:k_sim_user]
            
        top_k_indices = top_k_cache[u_idx]
        
        numerator = 0.0
        denominator = 0.0
        
        # 2. 用 top-k 相似用户的加权评分作为预测值
        for v_idx in top_k_indices:
            rating_vm = uti.values[v_idx, m_idx]
            if rating_vm > 0:
                sim_uv = sim_matrix[u_idx, v_idx]
                numerator += sim_uv * rating_vm
                denominator += sim_uv
        
        # 3. 若分母为 0，按原逻辑回退为 2.5
        if denominator == 0:
            pred_rating = 2.5
        else:
            pred_rating = numerator / denominator
            
        pred_rating = round(pred_rating, 4)
        predictions.append(pred_rating)
        
    # 4. 返回 numpy 数组
    return np.array(predictions)


def sse(predictions, ratings_test_set):
    # 计算 SSE = sum((pred - true)^2)
    true_ratings = ratings_test_set['rating'].values
    return np.sum((predictions - true_ratings) ** 2)


if __name__ == '__main__':
    sim_pearson = pearson_sim()
    # input p
    p = int(input())
    rec = recommend(p, sim_matrix=sim_pearson)
    print('recommended movies for ', p, ':')
    for i in range(len(rec)):
        print(rec[i][0], rec[i][1], rec[i][2], rec[i][3])

    predictions_pearson = prediction_test_set(sim_pearson, 10)
    print('sse_pearson:', sse(predictions_pearson, rating_test))