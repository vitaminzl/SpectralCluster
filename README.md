# 谱聚类代码

## 环境配置

本代码在python3.8环境下可成功运行。

在执行代码前，可先执行以下脚本，以安装所需的环境。

```shell
$ pip install -r requirements.txt
```



## PI

本文是对[Power iteration clustering](https://openreview.net/forum?id=SyWcksbu-H)

文章解读见[https://vitaminzl.com/2022/08/30/gnn/pin-pu-zai-ju-lei-zhong-de-ying-yong/#pi](https://vitaminzl.com/2022/08/30/gnn/pin-pu-zai-ju-lei-zhong-de-ying-yong/#pi)

运行前首先进入src文件夹

```shell
$ cd src
```

然后执行PI文件

```shell
$ python PI.py
```



## ROSC

本文是对文章[Rosc: Robust spectral clustering on multi-scale data](https://dl.acm.org/doi/abs/10.1145/3178876.3185993)的代码复现

文章解读见[https://vitaminzl.com/2022/08/30/gnn/pin-pu-zai-ju-lei-zhong-de-ying-yong/#rosc](https://vitaminzl.com/2022/08/30/gnn/pin-pu-zai-ju-lei-zhong-de-ying-yong/#rosc)

运行前首先进入src文件夹

```shell
$ cd src
```

然后执行ROSC文件

```shell
$ python ROSC.py
```

若想改变参数tknn、alpha1、alpha2的值，键入

```shell
$ python ROSC.py --tknn 8 --alpha1 1 --alpha2 0.01 
```

tknn、alpha1、alpha2的默认值分别为8、1、0.01



