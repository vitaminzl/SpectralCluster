import sklearn.datasets as ds
import numpy as np
import matplotlib.pyplot as plt


def get3CircleData(radius, nums, label_name=(1, 2, 3), noise=(0.03, 0.03, 0.03), seed=42):
    """
    该函数用于绘制3个大小不同的圆圈分布数据

    :param radius: (1, 3), 从小到大的圆圈半径
    :param nums: (1, 3), 不同半径数据对应的数量，
    :param noise: (1, 3), 不同半径数据对应的噪声
    :param seed: 随机数种子
    :return: data, label
    """
    data1, _ = ds.make_circles(n_samples=(nums[0], 0),
                               shuffle=True,
                               noise=noise[0],
                               random_state=seed)
    data2, _ = ds.make_circles(n_samples=(nums[1], 0),
                               shuffle=True,
                               noise=noise[1],
                               random_state=seed)
    data3, _ = ds.make_circles(n_samples=(nums[2], 0),
                               shuffle=True,
                               noise=noise[2],
                               random_state=seed)
    data1 *= [radius[0], radius[0]]
    data2 *= [radius[1], radius[1]]
    data3 *= [radius[2], radius[2]]
    data = np.array(np.vstack((data1, data2, data3)))
    label = np.zeros(data.shape[0], dtype=np.int32)
    label[:data1.shape[0]] = label_name[0]
    label[data1.shape[0]:(data1.shape[0] + data2.shape[0])] = label_name[1]
    label[(data1.shape[0] + data2.shape[0]):] = label_name[2]
    return data, label


def getDistanceSquare(X):
    n, m = X.shape
    G = np.dot(X, X.T)
    H = np.tile(np.diag(G), (n, 1))
    return H + H.T - 2 * G


def getSimilarMatrix(data, sigma):
    """
    根据高斯核公式获得相似度矩阵

    :param data: (N, M), N是数据的数量
    :param sigma: 放缩参数
    :return: S_mtx
    """
    S_mtx = np.exp(-getDistanceSquare(data) / (sigma ** 2 * 2))
    return S_mtx


def draw3CircleData(x, y, labels, title, subplt=False, sub_num=(1, 1, 1)):
    label_name = ["Class1", "Class2", "Class3"]
    if subplt:
        plt.subplot(sub_num[0], sub_num[1], sub_num[2])
    else:
        plt.figure()
    plt.scatter(x[labels == 1], y[labels == 1], color='red', s=4)
    plt.scatter(x[labels == 2], y[labels == 2], color='green', s=4)
    plt.scatter(x[labels == 3], y[labels == 3], color='blue', s=4)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.title(title)
    plt.legend(label_name, loc='upper right')


def PowerIter(W, iter_nums, labels, eps=1e-5, draw=False, draw_list=(0, 20, 60, 150)):
    """
    幂次迭代实验

    :param W: (N, N), 相似度矩阵
    :param iter_nums: (1,), 迭代次数
    :param labels: (N, ), 标签
    :param seed: 随机数种子
    :return: e_t
    """
    v_t = np.sum(W, axis=0) / np.sum(W)
    idx_list = np.array([i for i in range(labels.shape[0])])
    draw3CircleData(idx_list, v_t, labels, "Iterarion: " + str(0))
    pre_delta_t, delta_t = -10000, 10000
    eps = eps / W.shape[0]
    cnt = 1
    for i in range(iter_nums):
        v_t2 = W @ v_t
        v_t2 = v_t2 / np.sum(np.abs(v_t2))
        delta_t = np.sum(np.abs(v_t2 - v_t))
        if np.abs(pre_delta_t - delta_t) < eps:
            break
        pre_delta_t = delta_t
        v_t = v_t2
        if draw is True and i in draw_list:
            draw3CircleData(idx_list, v_t, labels,
                            "iteration: " + str(i + 1),
                            subplt=True,
                            sub_num=(2, len(draw_list) // 2, cnt))
            cnt += 1
    return v_t


def main():
    data, labels = get3CircleData(radius=[0.1, 6, 17], nums=[10, 30, 80])
    draw3CircleData(x=data[:, 0], y=data[:, 1], labels=labels, title="Data Set")
    S_mtx = getSimilarMatrix(data, sigma=1.8)
    W = np.diag(1 / np.sum(S_mtx, axis=0)) @ S_mtx
    v_t = PowerIter(W, iter_nums=300, eps=1e-5, labels=labels, draw=True)
    plt.show()


if __name__ == "__main__":
    main()
