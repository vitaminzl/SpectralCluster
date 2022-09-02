# 谱在聚类中的应用

本文主要对谱聚类的几篇论文进行解读，并对一部分的结果进行复现。本文首先从谱聚类的一般过程入手，介绍传统的谱聚类方法NCuts、NJW。针对传统方法的相似度量的缺陷，引入改进方法ZP；又针对特征向量的选择问题，引入改进方法PI。结合以上2种方式，加入TKNN，引入改进方法ROSC，接着对ROSC中修正相似度矩阵的缺陷，结合trace lasso正则项，引入改进方法CAST。最后，对于ROSC和CAST中都提到的Group Effect进行解读。文章结尾补充了幂代法和矩阵求导的内容。复现代码的仓库地址：[https://github.com/vitaminzl/SpectralCluster](https://github.com/vitaminzl/SpectralCluster) (备用镜像：[https://gitee.com/murphy_z/spectral-cluster](https://gitee.com/murphy_z/spectral-cluster))。



## Pipeline

![](https://imagehost.vitaminz-image.top/gnn-note-10.png)

谱聚类的一般过程如上图所示[^2]。对于一系列的数据，我们首先计算数据之间的相似度矩阵$S$，常用的相似度度量为高斯核$S_{ij}=\exp(-\frac{||\vec x_i-\vec x_j||^2}{2\sigma^2})$，然后求其拉普拉斯矩阵$L=D-S$，其中$D$为对角矩阵，且$D_{ii}=\sum A_{ij}$。然后求出拉普拉斯矩阵的特征向量，选择特征值$k$小的特征向量进行k-means聚类。

以上过程可以理解为，将原数据利用相似度度量转化为图数据，即使得每个数据间连着一条”虚边“，相似度即为边的权重。接下来将数据转换到频域上，选择一些低频作为数据的特征向量，这是因为低频成分往往具有更高层次、更具信息量的特征，而高频成分则更可能是噪声。然后对这些特征向量进行聚类。

而这种一般的方法在多尺度的数据聚类中往往表现不佳。



## NCuts

<img src="https://imagehost.vitaminz-image.top/li-spectral-cluster-4.png" style="zoom: 33%;" />

我们首先介绍一些远古的谱聚类方法。Normalized Cut[]



## NJW





## ZP

前面提到计算相似度矩阵时，我们常用高斯核$S_{ij}=\exp(-\frac{||\vec x_i-\vec x_j||^2}{2\sigma^2})$，而高斯核中$\sigma$的选取是需要考虑的，很多时候常常人工设定[^3]。但更重要的是$\sigma$是一个全局的参数，这在多尺度数据中具有一些缺陷。如设定的值较大时，稠密图的数据会趋于相似，设定较小时，稀疏图的数据则相似度过小。

![](https://imagehost.vitaminz-image.top/li-spectral-cluster-8.png)

ZP方法提出里一种局部调整$\sigma$的设定，距离度量修正为$S_{ij}=\exp(-\frac{||\vec x_i-\vec x_j||^2}{2\sigma_i\sigma_j})$。其中$\sigma_i, \sigma_j$是依附于$\vec x_i, \vec x_j$的是一种局部的参数。这一参数的设定应通过样本的特征取选取。在论文中[^3]中选择的方法是$\vec x_i$到第$K$个邻居的欧式距离，实验表明$K$取7在多个数据集上的效果表现良好。如上图所示，结点之间的边厚度表示数据间的权重大小（仅显示周围数据的权重）。图b是原来的高斯核距离度量，图c是修正后的。可以看到图b中靠近蓝点的边仍然比较厚，而图c则避免了的这种现象。



## PI

PI(Power Iteration)为幂迭代法[^5]。其灵感来源于幂迭代法用求主特征值（在[文章的后面部分](#Dominant Eigenvalue)会更详细地说明）。

我们设$W=D^{-1}A$，该矩阵有时候叫转移矩阵，因为它和马尔可夫的状态转移矩阵非常类似。每行的和为1，每个元素$W_{i,j}$可以看作是$i$结点到$j$结点转移的概率。它和归一化随机游走矩阵$L_r=I-W$有着重要的联系。NCuts算法证明了$L_r$第2小的特征值所对应的特征向量可以作为NCuts算法的一种近似。

这里需要说明的是，$L_r$最小的特征值为0，容易证明$\vec 1=[1, 1, ..., 1]$是$L_r$的0所对应的特征向量。而对于$W$来说则，其最大的特征值为1，且$\vec 1$是对应的特征向量。需要说明的是，$L_r$最小的几个特征向量或$W$最大的几个特征向量是有效的，其余可能是噪声。

首先给定一个向量$\vec v^{(0)}= c_1\vec  e_1 + c_2\vec e_2,..., +c_n\vec e_n$，其中$\vec e_i$为$W$的特征向量。且$\vec e_i$所对应的特征值$\lambda_i$满足$1=\lambda_1 > \lambda_2>...>\lambda_n$。

$$
\vec v^{(t+1)} = \frac{W\vec v^{(t)}}{||W\vec v^{(t)}||_1}
$$
假如我们按照如上的迭代公式进行迭代，则有如下过程（暂且忽略迭代公式的分母归一化项）
$$
\begin{align*}
\vec v^{(1)} &= W \vec v^{(0)} 
\\&=c_1W\vec  e_1 + c_2W\vec e_2,..., +c_nW\vec e_n
\\&=c_1\lambda_1  e_1 + c_2\lambda_2\vec e_2,..., +c_n\lambda_n\vec e_n
\end{align*}
$$
则
$$
\begin{align*}
\vec v^{(t)} &= Wv^{(t−1)} = W^2v^{(t−2)} = ... = W^tv^{(0)}
\\&=c_1W^t\vec  e_1 + c_2W^t\vec e_2,..., +c_nW^t\vec e_n
\\&=c_1\lambda_1^t  e_1 + c_2\lambda_2^t\vec e_2,..., +c_n\lambda_n^t\vec e_n
\\&=c_1\lambda_1^t\bigg[e_1+\sum\frac{c_2}{c_1}\bigg(\frac{\lambda_i}{\lambda_1}\bigg)^t\vec e_i)\bigg]
\end{align*}
$$
当$t\rightarrow +\infty$时，$\frac{c_2}{c_1}(\frac{\lambda_i}{\lambda_1})^t$会趋向于0。该方法的提出者认为，在有效成分$\vec e_i$的$\lambda_i$往往会接近于$\lambda_1$，而高频的一些噪声成分$\lambda_j$会接近于0。在迭代过程中使得有效成分$\vec e_i$前面的权重和噪声成分前的权重的差距会迅速扩大。

但是迭代的次数不宜过多，因为最后的结果会趋向于$k\vec 1$，因为$W$的主特征向量就是$\vec 1$。因此我们需设置一个迭代的门限值，以截断迭代过程。具体的算法如下。

<img src="https://imagehost.vitaminz-image.top/gnn-note-11.png" style="zoom: 40%;" />

这里以论文中开始提到的3圆圈数据集为例子，进行结果的复现，如下图所示。

<img src="https://imagehost.vitaminz-image.top/li-spectral-cluster-5.png" style="zoom:50%;" />

通过以上的算法，我选取几次的迭代结果$\vec v^{(t)}$进行可视化，同一个类别在后面的迭代过程中逐渐局部收敛到一个值。

<img src="https://imagehost.vitaminz-image.top/li-spectral-cluster-7.png" style="zoom: 60%;" />

在实验中发现，其结果和许多因素有关，其中包括高斯核距离度量中的$\sigma$，初始的向量$\vec v^{(0)}$（论文中取$\vec v^{(0)}= \frac{\sum_j A_{ij}}{\sum_i\sum_j A_{ij}}$），结束时的$\hat\epsilon$设置，甚至发现使用$W^T$具有更好的效果，这是因为$W^T$的主特征值的特征向量就已经具有分类的效果（其意义尚待研究），而$W$的主特征向量是$\vec 1$，但这仅仅针对于3圆圈这一数据集而言。在[问题与总结](#问题与总结)中，会提到这一点。此外还有计算机的运算精度也会影响结果。

该方法的一个重要优点是简单高效，其收敛速度快，在百万级别的数据中也能在几秒内收敛。缺陷是过分拔高了特征值大的部分，在多尺度数据中存在一些低特征值但仍然重要的信息。

以下是主函数的代码，完整代码见：https://github.com/vitaminzl/SpectralCluster/blob/master/PI.py

```python
def main():
    data, labels = get3CircleData(radius=[0.1, 6, 17], nums=[10, 30, 80])
    draw3CircleData(x=data[:, 0], y=data[:, 1], labels=labels, title="Data Set")
    S_mtx = getSimilarMatrix(data, sigma=1.8)
    W = np.diag(1 / np.sum(S_mtx, axis=0)) @ S_mtx
    v_t = PowerIter(W, iter_nums=300, eps=1e-5, labels=labels)
    plt.show()
```



## TKNN

![](https://imagehost.vitaminz-image.top/li-spectral-cluster-10.png)

聚类问题转化为图问题时，需要解决邻接问题。定义结点之间的连接常常有2种方式[^4]。第一种如上图左，每个数据选择自己最近的K个邻居相邻接，得到的图被称为K邻接图（K Nearest Neighbor Graph）；第二种如上图右，每个结点选择半径$\epsilon$的邻居相邻接。

![](https://imagehost.vitaminz-image.top/li-spectral-cluster-14.png)

假如我们使用KNN的方法，K取4。如上图a所示，红色结点的4个最近邻用红线连接，绿色结点的4个最近邻用绿线连接。我们会发现，虽然红色的最近邻包括绿色，但绿色不包括红色，我们称红色和绿色不是**相互近邻**。但如图b所示，则红色和绿色则为相互近邻。若2个结点是相互近邻，则称这2个结点**相互可达**。如图c所示，红色与绿色是相互近邻，绿色和黄色相互近邻，那么红色和绿色也**相互可达**。

Transitive K Nearest Neighbor(TKNN) Graph 是指当2个结点时相互可达的，则二者连接一条边。因此其邻接矩阵$W$中，若2个点$i,j$相互可达，$W_{i,j}=W_{j,i}=1$。

构造的过程可描述如下：

* step1: 构造K邻接矩阵$A$，对于结点$i$由$k$个邻居$j$，则$A_{i, j}=1$
* step2: 构造相互近邻矩阵$A'=A A^T$，若为相互近邻，则为1，否则为0。
* step3: 寻找$A'$的所有连通分量$S$
* step4: 对于连通分量$S_i$中的每2个元素$S_{i,j}, S_{i,k}$，令$W_{S_{i,j},S_{i,k}}=W_{S_{i,k},S_{i,j}}=1$，其余为0。

代码如下：

```python
def getTKNN_W():
    
```





## ROSC

![](https://imagehost.vitaminz-image.top/li-spectral-cluster-9.png)

如上图所示为ROSC方法的流程图。ROSC方法[^1]结合了以上2种方法，但相比于PI方法不同的是，它并不是将PI得到的输出直接作为k-means聚类的输入，而是增加了一个求修正相似矩阵的过程。

首先随机设置不同的$\vec v^{(0)}$获取$p$个“伪特征向量”，拼成一个$p\times n$的矩阵矩阵$X$，并对$X$进行标准化，即使得$XX^T=I$。

ROSC论文中认为，相似度矩阵的意义可以表示为某一个实体能够被其他实体所描述的程度。即任何一个实体$x_{i}=\sum Z_{i,j}x_j$，这里$Z_{i,j}$即为修正相似度矩阵。因此就有：
$$
X=XZ+O
$$
其中$O$表示噪声矩阵。

定义优化问题
$$
\min_{Z}||X-XZ||_F^2+\alpha_1||Z||_F+\alpha_2||W-Z||_F
$$
优化问题的第一项表示最小化噪声，第二项则是$Z$的Frobenius 范数[^12]），为正则化项，用于平衡其他2项，第三项则是减小与前文中TKNN的邻接矩阵$W$的差距。$\alpha_1,\alpha_2$是平衡参数，需要人工设置。

求解以上优化问题，可以先对$Z$求导（[文章的后面](#Derivatives of Matrix)还会做一些补充），使导数为0即可。对三项项求导有
$$
\begin{align*}
\frac{\partial ||X-XZ||^2_F}{\partial Z}&=-2X^T(X-XZ)\\
\frac{\partial\alpha_1||Z||^2_F}{\partial Z}&=2\alpha_1Z\\
\frac{\partial\alpha_2||W-Z||^2_F}{\partial Z}&=-2\alpha_2(W-Z)
\end{align*}
$$
三项相加有
$$
-X^T(X-XZ)+\alpha_1Z-\alpha_2(W-Z)=0
$$
整理可得
$$
Z^*=(2X^TX+\alpha_1 I+\alpha_2 I)^{-1}(X^TX+\alpha_2W)
$$
但这样求出来的$Z^*$可能使不对称的，且可能存在负数。所以这里再次做了一个修正$\tilde Z=(|Z^*|+|Z^*|^T)/2$。

接下来就可以执行一般的谱聚类方法了。具体的算法流程可以用如下图所示：

<img src="https://imagehost.vitaminz-image.top/li-spectral-cluster-12.png" style="zoom: 33%;" />

算法第4行中的whiten为白化处理[^15]，是数据预处理的一种常用方法。它类似于PCA，但与PCA不同的是，PCA往往用来降维，而白化则是利用PCA的特征向量，将数据转换到新的特征空间，然后对新的坐标进行方差归一化，目的是去除输入数据的冗余信息。

```python
def ROSC(data):
    
    
```





## Trace Lasso

Trace Lasso[^13]是一种介于L1和L2正则的正则化项。其形式为
$$
\Omega(W)=||XDiag(W)||_*
$$
其中$X$为已归一化的特征矩阵，即对于特征$\vec x_i$有$\vec x_i \vec x_i^T=1$。$W$为待求参数。$||·||_*$为核范数[^14]（或迹范数）,$||Z||_*=tr(\sqrt{Z^TZ})$，表示所有奇异值之和。

Trace Lasso具有如下性质：

* 当$X^TX=I$时，即特征之间的相关性为0，或者正交，那么
  $$
  \Omega(W)=||W||_1
  $$
  即退化为1范式。

* 当$X^TX=\vec 1^T\vec 1$时，即所有特征都完全相关，那么
  $$
  \Omega(W)=||W||_2
  $$
  即退化为2范式

* 其他情况下，在1范式和2范式之间。

Trace Lasso的优点就是它可以根据数据的特征，接近合适的范式，这相比弹性网络更好。



## CAST

ROSC方法虽然可以加强类内数据的联系，但没有使得类间的间距增大。

而CAST相比于ROSC的区别就在于修改了优化函数，成为如下形式：
$$
\min_{Z} \frac{1}{2} ||\vec x-X\vec z||_2+\alpha_1||XDiag(\vec z)||_*+\frac{\alpha_2}{2}||W-\vec z||_2
$$
其中$\vec x$是$X$的其中一个特征向量，$z$是修正相似度矩阵$Z$中的一个向量。于ROSC的主要区别在于范数的选择，通过trace lasso可以使得其具有类内聚合也有类外稀疏的特性。

该优化问题的求解已经超出了我的能力范围，在此直接贴出论文中的算法流程

<img src="https://imagehost.vitaminz-image.top/li-spectral-cluster-15.png" style="zoom: 50%;" />

整个CAST算法的流程如下：

<img src="https://imagehost.vitaminz-image.top/li-spectral-cluster-16.png" style="zoom:50%;" />

对比ROSC，他们的区别就在于求解$Z^*$的方法不同，其余都是一样的。

```python
def CAST(data):
    
```





## Group Effect





## 补充

### Dominant Eigenvalue





### Derivatives of Matrix 

矩阵求导是矩阵论中的相关知识[^11]，这里仅对前文用到的Frobenius范式矩阵的求导过程进行介绍。

Frobenius范式可以表示成如下的迹形式
$$
||X||_F=\sqrt{tr(X^TX)}
$$
首先引入2个求导法则

* 法则1：若$A,X$为$m\times n$的矩阵，有
  $$
  \frac{\partial tr(A^TX)}{\partial X}=\frac{\partial tr(X^TA)}{\partial X}=A
  $$

* 法则2：若$A$为$m\times m$的矩阵，$X$为$m\times n$的矩阵，有

$$
\frac{\partial tr(X^TAX)}{\partial X}=AX+A^TX
$$

因此若求以下导数
$$
\frac{\partial ||A-BX||^2_F}{\partial X}
$$
利用以上法则有：
$$
\begin{align*}
\frac{\partial||A-BX||^2_F}{\partial X}&=\frac{\partial tr[(A-BX)^T(A-BX)]}{\partial X}\\
&=\frac{\partial tr[(A^T-X^TB^T)(A-BX)]}{\partial X}\\
&=\frac{\partial[tr(A^TA)-2tr(A^TBX)+tr(X^TB^TBX)]}{\partial X}\\
&=0-2A^TB+B^TBX+B^TBX\\
&=-2B^T(A+BX)
\end{align*}
$$



## 问题与总结





## 参考资料

[^1]: [Li X, Kao B, Luo S, et al. Rosc: Robust spectral clustering on multi-scale data[C]//Proceedings of the 2018 World Wide Web Conference. 2018: 157-166.](https://dl.acm.org/doi/abs/10.1145/3178876.3185993)
[^2]: [Li X, Kao B, Shan C, et al. CAST: a correlation-based adaptive spectral clustering algorithm on multi-scale data[C]//Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020: 439-449.](https://dl.acm.org/doi/abs/10.1145/3394486.3403086)
[^3]: [Zelnik-Manor L, Perona P. Self-tuning spectral clustering[J]. Advances in neural information processing systems, 2004, 17.](https://proceedings.neurips.cc/paper/2004/hash/40173ea48d9567f1f393b20c855bb40b-Abstract.html)
[^4]: [https://csustan.csustan.edu/~tom/Clustering/GraphLaplacian-tutorial.pdf](https://csustan.csustan.edu/~tom/Clustering/GraphLaplacian-tutorial.pdf)
[^5]:[Lin F, Cohen W W. Power iteration clustering[C]//ICML. 2010.](https://openreview.net/forum?id=SyWcksbu-H)
[^6]: [Li Z, Liu J, Chen S, et al. Noise robust spectral clustering[C]//2007 IEEE 11th International Conference on Computer Vision. IEEE, 2007: 1-8.](https://ieeexplore.ieee.org/abstract/document/4409061)
[^7]: [Meilă M, Shi J. A random walks view of spectral segmentation[C]//International Workshop on Artificial Intelligence and Statistics. PMLR, 2001: 203-208.](https://proceedings.mlr.press/r3/meila01a.html)
[^8]: [Shi J, Malik J. Normalized cuts and image segmentation[J]. IEEE Transactions on pattern analysis and machine intelligence, 2000, 22(8): 888-905.](https://ieeexplore.ieee.org/abstract/document/868688)
[^9]: https://zhuanlan.zhihu.com/p/336250805
[^10]: [Zelnik-Manor L, Perona P. Self-tuning spectral clustering[J]. Advances in neural information processing systems, 2004, 17.](https://proceedings.neurips.cc/paper/2004/hash/40173ea48d9567f1f393b20c855bb40b-Abstract.html)
[^11]: [Petersen K B, Pedersen M S. The matrix cookbook[J]. Technical University of Denmark, 2008, 7(15): 510.](https://ece.uwaterloo.ca/~ece602/MISC/matrixcookbook.pdf)

https://www.cs.huji.ac.il/w~csip/tirgul2.pdf

[^12]: [https://mathworld.wolfram.com/FrobeniusNorm.html](https://mathworld.wolfram.com/FrobeniusNorm.html)

[^13]: [Grave E, Obozinski G R, Bach F. Trace lasso: a trace norm regularization for correlated designs[J]. Advances in Neural Information Processing Systems, 2011, 24.](https://proceedings.neurips.cc/paper/2011/hash/33ceb07bf4eeb3da587e268d663aba1a-Abstract.html)
[^14]: [https://en.wikipedia.org/wiki/Matrix_norm](https://en.wikipedia.org/wiki/Matrix_norm)
[^15]: [https://en.wikipedia.org/wiki/Whitening_transformation](https://en.wikipedia.org/wiki/Whitening_transformation)
