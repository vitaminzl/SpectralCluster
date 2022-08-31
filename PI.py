import sklearn.datasets as ds
import numpy as np
import matplotlib.pyplot as plt


def get3CircleData(radius, nums, noise=(0.03, 0.03, 0.03), seed=42):
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
    label[data1.shape[0]:(data1.shape[0] + data2.shape[0])] = 1
    label[(data1.shape[0] + data2.shape[0]):] = 2
    return data, label


def getSimilarMatrix(data, sigma):
    S_mtx = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            S_mtx[i, j] = np.exp(-np.sum((data[i, :] - data[j, :]) ** 2) / (sigma ** 2 * 2))
    return S_mtx


def draw3CircleData(x, y, labels, title):
    fig, ax = plt.subplots()
    label_name = ["Class1", "Class2", "Class3"]
    ax.scatter(x[labels == 0], y[labels == 0], color='red'
               , s=40)
    ax.scatter(x[labels == 1], y[labels == 1], color='green'
               , s=40)
    ax.scatter(x[labels == 2], y[labels == 2], color='blue',
               s=40)
    ax.set(xlabel='X',
           ylabel='Y',
           title=title)
    ax.legend(label_name, loc='upper right')
    plt.show()


def PowerIter(W, iter_nums, labels, seed=42):
    e_t = np.random.randn(W.shape[0], 1)
    idx_list = np.array([i for i in range(labels.shape[0])])
    draw3CircleData(idx_list, e_t, labels, "Iterarion: " + str(0))
    for i in range(iter_nums):
        e_t = W @ e_t
        e_t = e_t / np.sum(np.abs(e_t))
        if (i + 1) % 10 == 0:
            draw3CircleData(idx_list, e_t, labels, "Iterarion: " + str(i+1))
            # print(e_t)

    return e_t


def main():
    data, labels = get3CircleData(radius=[0.1, 1, 3], nums=[10, 30, 80])
    draw3CircleData(x=data[:, 0], y=data[:, 1], labels=labels, title="Data Set")
    S_mtx = getSimilarMatrix(data, sigma=1)
    W = S_mtx / np.sum(S_mtx, axis=0)
    e_t = PowerIter(W, iter_nums=100, labels=labels)


if __name__ == "__main__":
    main()


