#

import numpy as np
import matplotlib.pyplot as plt

def DTW(s1, s2):
    #
    if len(s1) == len(s2):
        n = len(s1)
    else:
        return
    # 构建二位dp矩阵,存储对应每个子问题的最小距离
    dp = np.zeros(shape=[n, n])
    # 起始条件,计算单个字符与一个序列的距离
    dis_lambda = lambda x, y: np.abs(x-y)
    for i in range(n):
        dp[i][0] = dis_lambda(s1[i], s2[0])
    for j in range(n):
        dp[0][j] = dis_lambda(s1[0], s2[j])
    # 利用递推公式,计算每个子问题的最小距离,矩阵最右下角的元素即位最终两个序列的最小值
    for i in range(1, n):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + dis_lambda(s1[i], s2[j])
    return dp[-1][-1]

def cal_cos_dis(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def weight_matrix(i_1, j_1, i_2, j_2, sigma2=10, epsilon=0.5):
    d = np.sqrt(np.square(i_1 - i_2) + np.square(j_1 - j_2))
    return np.exp(-d / sigma2) * (np.exp(-d / sigma2) >= epsilon)



flow_cube = np.load('flow_real_big.npy')
flow_cube_ = np.reshape(flow_cube, newshape=[24*6, 61, 10, 10])
flow_cube_mean = np.mean(flow_cube_, axis=1)
flow_cube_mean = np.reshape(flow_cube_mean, newshape=[144, 100])
flow_cube_norm = flow_cube_mean / np.max(flow_cube_mean)

# 余弦相似度 或 DTW相似度
dis_sectors = np.zeros(shape=[100, 100])
for i in range(100):
    print("Processing: ", i)
    for j in range(i+1, 100):
        # dis_sectors[i][j] = DTW(flow_cube_norm[:, i], flow_cube_norm[:, j])
        dis_sectors[i][j] = cal_cos_dis(flow_cube_norm[:, i], flow_cube_norm[:, j])
        dis_sectors[j][i] = dis_sectors[i][j]

plt.figure()
plt.imshow(dis_sectors)
plt.show()

# np.save('WeightsGraphCos.npy', dis_sectors)


# 距离
dis_sectors = np.zeros(shape=[100, 100])

for i in range(100):
    i_1, j_1 = i // 10, i % 10
    for j in range(100):
        if i == j:
            continue
        i_2, j_2 = j // 10, j % 10
        print(i_1, j_1, i_2, j_2)
        dis_sectors[i][j] = weight_matrix(i_1, j_1, i_2, j_2)


plt.figure()
plt.imshow(dis_sectors)
plt.show()
print(np.max(dis_sectors))
np.save('WeightsGraphDis.npy', dis_sectors)
