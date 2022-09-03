import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import rand_score


def getPurityScore(label_true, label_pred):
    y_voted_labels = np.zeros(label_true.shape)
    labels = np.unique(label_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        label_true[label_true == labels[k]] = ordered_labels[k]
    labels = np.unique(label_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(label_pred):
        hist, _ = np.histogram(label_true[label_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[label_pred == cluster] = winner

    return accuracy_score(label_true, y_voted_labels)


def assess(label_true, label_pred):
    purity = getPurityScore(label_true, label_pred)
    AMI = adjusted_mutual_info_score(label_true, label_pred)
    RI = rand_score(label_true, label_pred)

    return purity, AMI, RI


def draw(x, y, label):
    label_uni = list(set(label))
    color_list = ["red", "green", "blue"]
    for i in range(len(label_uni)):
        plt.scatter(x[label_uni[i] == label], y[label_uni[i] == label], color=color_list[i], s=4)

    # plt.xlabel('X')
    # plt.ylabel('Y')
    plt.legend(label_uni, loc='upper right')
    plt.show()


def ncuts(L, K, seed=42, eps=1e-15):
    D = np.diag(1 / (np.sqrt(np.sum(L, axis=0)) + eps))
    L = D @ L @ D
    eig_val, eig_vec = np.linalg.eigh(L)
    idx = np.argsort(-eig_val)
    eig_vec = eig_vec[:, idx]
    eig_vec = eig_vec[:, 1:K]
    eig_vec = D @ eig_vec
    C = KMeans(n_clusters=K, random_state=seed).fit(eig_vec)
    return C.labels_
