import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minihash import miniHash

# ---------- 固定参数 ----------
USER_ID = 12
K_NEIGHBORS = 10
# -----------------------------

print("=" * 50)
print("开始加载数据集...")

# 1. 加载电影数据（用于 CF 任务，仅索引）
movies = pd.read_csv('datasets/movies.csv', index_col=0)

# 2. 加载训练集，构建评分字典和二值化字典
rating_dic = {}
rating_dic_01 = {}
with open('datasets/train_set.csv', 'r', encoding='UTF-8') as f:
    lines = f.readlines()[1:]
    for line in lines:
        item = line.strip().split(',')
        uid, mid, r = item[0], item[1], float(item[2])
        if uid not in rating_dic:
            rating_dic[uid] = {}
        rating_dic[uid][mid] = r
        if uid not in rating_dic_01:
            rating_dic_01[uid] = {}
        rating_dic_01[uid][mid] = 1 if r >= 3.0 else 0

uti = pd.DataFrame(rating_dic, dtype='float').T.fillna(0)          # 原始评分矩阵
uti_jaccard = pd.DataFrame(rating_dic_01).T.fillna(0).astype(int) # 二值化矩阵

# 3. 加载测试集
rating_test = pd.read_csv('datasets/test_set.csv')
print(f"用户数: {uti.shape[0]}, 电影数: {uti.shape[1]}")
print("=" * 50)

# ---------- 通用 SSE 计算函数 ----------
def calc_sse(preds):
    true = rating_test['rating'].values
    return np.sum((preds - true) ** 2)

# ---------- 任务一：Pearson 相似度（基线） ----------
def pearson_sim():
    sim = np.corrcoef(uti.values)
    return np.nan_to_num(sim)

def prediction_test_set_cf(sim_matrix, k=10):
    user_list = list(uti.index)
    preds = []
    for _, row in rating_test.iterrows():
        u = str(int(row['userId']))
        i = str(int(row['movieId']))
        if u not in user_list or i not in uti.columns:
            preds.append(2.5)
            continue
        u_idx = user_list.index(u)
        sims = pd.Series(sim_matrix[u_idx], index=user_list)
        sims = sims.drop(u)
        top_k = sims.nlargest(k, keep='first')
        num, den = 0.0, 0.0
        for v, s in top_k.items():
            r = uti.at[v, i]
            if r > 0:
                num += s * r
                den += s
        pred = 2.5 if den == 0 else num / den
        preds.append(round(pred, 4))
    return np.array(preds)

# 计算任务一 SSE
sim_pearson = pearson_sim()
preds_pearson = prediction_test_set_cf(sim_pearson, K_NEIGHBORS)
sse_pearson = calc_sse(preds_pearson)
print(f"[基线] UserCF-Pearson SSE = {sse_pearson:.4f}")

# ---------- 任务二：Content-Cosine（基线） ----------
# 需要加载 movies 并计算 TF-IDF 特征（复用 Content_based_rec 中的函数）
from Content_based_rec import tokenize, featurize, get_feature_matrix2, get_cosine_sim
movies_cb = pd.read_csv('datasets/movies.csv')
movies_cb = tokenize(movies_cb)
movies_cb, vocab = featurize(movies_cb)
feature_matrix = get_feature_matrix2(movies_cb, vocab)
sim_cosine = get_cosine_sim(feature_matrix)
movies_map = {row['movieId']: idx for idx, row in movies_cb.iterrows()}

def prediction_test_set_cb(sim_matrix):
    # 构建用户历史
    train_df = pd.read_csv('datasets/train_set.csv')
    user_history = {}
    for _, row in train_df.iterrows():
        u, m, r = row['userId'], row['movieId'], row['rating']
        user_history.setdefault(u, []).append((m, r))
    preds = []
    for _, row in rating_test.iterrows():
        u, i = row['userId'], row['movieId']
        history = user_history.get(u, [])
        if not history:
            preds.append(2.5)
            continue
        mean_r = sum(r for _, r in history) / len(history)
        idx_i = movies_map.get(i)
        if idx_i is None:
            preds.append(mean_r)
            continue
        num, den = 0.0, 0.0
        for j, rj in history:
            idx_j = movies_map.get(j)
            if idx_j is not None:
                sim = sim_matrix[idx_i, idx_j]
                if sim > 0:
                    num += sim * rj
                    den += sim
        preds.append(num / den if den > 0 else mean_r)
    return np.array(preds)

preds_cosine = prediction_test_set_cb(sim_cosine)
sse_cosine = calc_sse(preds_cosine)
print(f"[基线] Content-Cosine SSE = {sse_cosine:.4f}")

# ---------- 任务三、四：MinHash 变体 ----------
def prediction_test_set_cf_minihash(sim_matrix):
    # 完全复用上面的 CF 预测，但 sim_matrix 来自 MinHash
    return prediction_test_set_cf(sim_matrix, K_NEIGHBORS)

def prediction_test_set_cb_minihash(sim_matrix):
    # 完全复用上面的 CB 预测，但 sim_matrix 来自 MinHash
    return prediction_test_set_cb(sim_matrix)

# ---------- 准备 Content 的 0-1 特征矩阵（用于 CB-MinHash） ----------
# 注意：我们需要的是电影 × token 的 0-1 矩阵，直接复用 Content_based_minihash 中的函数
from Content_based_minihash import tokenize as tokenize_cb, featurize as featurize_cb, get_feature_matrix1
movies_cb_mini = pd.read_csv('datasets/movies.csv')
movies_cb_mini = tokenize_cb(movies_cb_mini)
movies_cb_mini, vocab_cb = featurize_cb(movies_cb_mini)
feature_matrix_01 = get_feature_matrix1(movies_cb_mini, vocab_cb)  # (电影数, token数)

# ---------- 实验不同 H ----------
hash_counts = [2, 5, 10, 20, 40, 80, 100]
sse_cf_minihash = []
sse_cb_minihash = []

print("\n开始 MinHash 哈希数量实验...")
for h in hash_counts:
    print(f"\n>>> 测试 H = {h}")
    
    # CF-MinHash
    sim_cf = miniHash(uti_jaccard.T, h)
    preds_cf = prediction_test_set_cf_minihash(sim_cf)
    sse_cf_val = calc_sse(preds_cf)
    sse_cf_minihash.append(sse_cf_val)
    print(f"  CF-MinHash SSE = {sse_cf_val:.4f}")
    
    # CB-MinHash
    sim_cb = miniHash(feature_matrix_01.T, h)   # 转置使列为电影
    preds_cb = prediction_test_set_cb_minihash(sim_cb)
    sse_cb_val = calc_sse(preds_cb)            # 现在直接调用 calc_sse，不再用未定义的 sse_cb
    sse_cb_minihash.append(sse_cb_val)
    print(f"  CB-MinHash SSE = {sse_cb_val:.4f}")

# ---------- 绘图（英文标签避免字体问题） ----------
plt.figure(figsize=(10, 6))

# 两条曲线
plt.plot(hash_counts, sse_cf_minihash, marker='o', label='UserCF + MinHash', linewidth=2)
plt.plot(hash_counts, sse_cb_minihash, marker='s', label='Content + MinHash', linewidth=2)

# 两条水平参考线（基线）
plt.axhline(y=sse_pearson, color='#1f77b4', linestyle='--', linewidth=1.5, label=f'UserCF-Pearson (baseline) = {sse_pearson:.2f}')
plt.axhline(y=sse_cosine, color='#2ca02c', linestyle='--', linewidth=1.5, label=f'Content-Cosine (baseline) = {sse_cosine:.2f}')

plt.xlabel('Number of Hash Functions (H)', fontsize=12)
plt.ylabel('SSE (Sum of Squared Errors)', fontsize=12)
plt.title('Impact of MinHash Hash Count on SSE (with Baselines)', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

# 在曲线点上标数值
for x, y in zip(hash_counts, sse_cf_minihash):
    plt.text(x, y + 0.5, f'{y:.2f}', ha='center', va='bottom', fontsize=8)
for x, y in zip(hash_counts, sse_cb_minihash):
    plt.text(x, y + 0.5, f'{y:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('hash_effect_with_baselines.png', dpi=300)
plt.show()

print("\n实验完成！折线图已保存为 hash_effect_with_baselines.png")
print("图中包含两条基线（水平虚线）和两条 MinHash 曲线。")