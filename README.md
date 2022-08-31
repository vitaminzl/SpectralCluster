# 频谱在聚类中的应用

本文主要对2篇论文

https://github.com/vitaminzl/SpectralCluster (备用镜像：https://gitee.com/murphy_z/spectral-cluster)

现有的频谱聚类（Spectral Cluster）的方法在多尺度数据表现不佳。

相似矩阵->拉普拉斯矩阵->特征向量->聚类方法

2种方法：一种是适当地放缩相似度矩阵，另一种伪特征向量。

高斯核计算相似矩阵$S_{ij}=\exp(\frac{||\vec x_i-\vec x_j||^2}{2\sigma^2})$





## Pipeline







## ZP





## PI





## TKNN





## ROSC





## CAST





## Group Effect





## 补充

### Normalized Cuts



<img src="https://imagehost.vitaminz-image.top/li-spectral-cluster-4.png" style="zoom: 33%;" />



### Power Method





## 总结





## 参考资料

[^1]: [Li X, Kao B, Luo S, et al. Rosc: Robust spectral clustering on multi-scale data[C]//Proceedings of the 2018 World Wide Web Conference. 2018: 157-166.](https://dl.acm.org/doi/abs/10.1145/3178876.3185993)
[^2]: [Li X, Kao B, Shan C, et al. CAST: a correlation-based adaptive spectral clustering algorithm on multi-scale data[C]//Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020: 439-449.](https://dl.acm.org/doi/abs/10.1145/3394486.3403086)
[^3]: [Belkin M, Niyogi P. Laplacian eigenmaps and spectral techniques for embedding and clustering[J]. Advances in neural information processing systems, 2001, 14.](https://proceedings.neurips.cc/paper/2001/hash/f106b7f99d2cb30c3db1c3cc0fde9ccb-Abstract.html)
[^6]: https://csustan.csustan.edu/~tom/Clustering/GraphLaplacian-tutorial.pdf
[^5]:[Lin F, Cohen W W. Power iteration clustering[C]//ICML. 2010.](https://openreview.net/forum?id=SyWcksbu-H)
[^6]: [Li Z, Liu J, Chen S, et al. Noise robust spectral clustering[C]//2007 IEEE 11th International Conference on Computer Vision. IEEE, 2007: 1-8.](https://ieeexplore.ieee.org/abstract/document/4409061)
[^7]: [Meilă M, Shi J. A random walks view of spectral segmentation[C]//International Workshop on Artificial Intelligence and Statistics. PMLR, 2001: 203-208.](https://proceedings.mlr.press/r3/meila01a.html)
[^8]: [Shi J, Malik J. Normalized cuts and image segmentation[J]. IEEE Transactions on pattern analysis and machine intelligence, 2000, 22(8): 888-905.](https://ieeexplore.ieee.org/abstract/document/868688)
[^9]: https://zhuanlan.zhihu.com/p/336250805
