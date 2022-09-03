import numpy as np


def whiten(X, eps=1e-15):
    # 减去均值，使得以0为中心
    X -= np.mean(X, axis=0)
    # 计算协方差矩阵
    cov = np.dot(X.T, X) / X.shape[0]
    U, S, V = np.linalg.svd(cov)
    XRot = np.dot(X, U)
    XWhite = XRot / np.sqrt(S + eps)
    return XWhite


def getDistanceSquare(X):
    n, m = X.shape
    G = np.dot(X, X.T)
    H = np.tile(np.diag(G), (n, 1))
    return H + H.T - 2 * G


def getSimilarMatrix(data, sigma):
    S_mtx = np.exp(-getDistanceSquare(data) / (sigma ** 2 * 2))
    return S_mtx


def getSimilarMatrix2(data, k=7, eps=1e-15):
    N = data.shape[0]
    dist_square = getDistanceSquare(data)
    idx_s = np.argsort(dist_square, axis=0)
    sigma = np.sqrt(dist_square[np.arange(N), idx_s[:, k]]).reshape(N, 1)
    sigma = sigma @ sigma.T
    S_mtx = np.exp(-dist_square / (sigma + eps))
    return S_mtx


def PIC_k(W, k, iter_nums=1000, eps=1e-5):
    N = W.shape[0]
    pre_delta_t, delta_t = -10000, 10000
    eps = eps / N
    v_tk = np.zeros((N, k))
    for j in range(k):
        v_t = np.random.rand(N, 1)
        for i in range(iter_nums):
            v_t2 = W @ v_t
            v_t2 = v_t2 / np.sum(np.abs(v_t2))
            delta_t = np.sum(np.abs(v_t2 - v_t))
            if np.abs(pre_delta_t - delta_t) < eps:
                break
            pre_delta_t = delta_t
            v_t = v_t2
        v_tk[:, j] = v_t.reshape(-1)

    return v_tk


def norm(X, eps=1e-15):
    return X / np.sqrt(np.sum(X ** 2, axis=0) + eps)
