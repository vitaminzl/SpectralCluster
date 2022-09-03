import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import preprocess as prep
import postprocess as postp
import argparse


def getTKNN_W(S_mtx, K):
    N = S_mtx.shape[0]
    # 构造KNN矩阵
    KNN_A = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        idx = np.argsort(S_mtx[i, :])
        KNN_A[i, idx[(N - K):]] = 1
    MKNN_A = KNN_A * KNN_A.T
    G = nx.from_numpy_array(MKNN_A)
    compo_list = [c for c in nx.connected_components(G)]
    # 构造TKNN矩阵
    TKNN_W = np.zeros((N, N), dtype=np.int32)
    for c_i in compo_list:
        c = np.array(list(c_i), dtype=np.int32)
        idx_c = np.tile(c, (len(c), 1))
        TKNN_W[idx_c.T, idx_c] = 1 - np.identity(len(c))

    return TKNN_W


def getROSC_Z(X, W, alpha1, alpha2):
    N = X.shape[1]
    I = np.identity(N)
    Z = np.linalg.inv(X.T @ X + alpha1 * I + alpha2 * I)
    Z = Z @ (X.T @ X + alpha2 * W)
    return Z


def ROSC(S, C_k, t_k, alpha1, alpha2):
    W_tknn = getTKNN_W(S, K=t_k)
    W = np.diag(np.sum(S, axis=0)) @ S
    X = prep.PIC_k(W, k=C_k)
    X = prep.whiten(X)
    X = prep.norm(X)
    Z = getROSC_Z(X.T, W_tknn, alpha1, alpha2)
    Z = (np.abs(Z) + np.abs(Z.T)) / 2
    C = postp.ncuts(Z, C_k)
    return C


def main(data_name, t_k=8, alpha1=1, alpha2=0.01):
    data = np.loadtxt("../dataset/" + data_name + ".txt", delimiter=',', dtype=np.float64)
    label = np.loadtxt("../dataset/" + data_name + "Label.txt", dtype=np.int32)
    C_k = len(set(label))
    # postp.draw(data[:, 0], data[:, 1], label)
    S = prep.getSimilarMatrix2(data=data)
    # max_t = 12
    # prt_list = np.zeros(max_t)
    # AMI_list = np.zeros(max_t)
    # RI_list = np.zeros(max_t)
    # for t in range(max_t):
    C = ROSC(S, C_k=C_k, t_k=t_k, alpha1=alpha1, alpha2=alpha2)
    prt, AMI, RI = postp.assess(label_true=label, label_pred=C)
    # prt_list[t] = prt
    # AMI_list[t] = AMI
    # RI_list[t] = RI

    # plt.plot(1 + np.arange(max_t), prt_list)
    # plt.plot(1 + np.arange(max_t), AMI_list)
    # plt.plot(1 + np.arange(max_t), RI_list)
    # plt.legend(["Purity", "AMI", "RI"])
    # plt.show()

    print(f"{data_name}\nPurity: {prt}\nAMI: {AMI}\nRI: {RI}\n")
    # postp.draw(data[:, 0], data[:, 1], C)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spectral Cluster')
    parser.add_argument('--tknn', type=int, default=8)
    parser.add_argument('--alpha1', type=float, default=1)
    parser.add_argument('--alpha2', type=float, default=0.01)
    args = parser.parse_args()

    dataset = ["Syn", "COIL20", "Glass", "Isolet", "Mnist0127", "Yale"]
    for data_name in dataset:
        main(data_name=data_name, t_k=args.tknn, alpha1=args.alpha1, alpha2=args.alpha2)
