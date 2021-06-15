# crypten_vfl_demo

使用crypten做纵向联邦学习demo。详细介绍参考[知乎专栏](https://zhuanlan.zhihu.com/p/364257618)

## 数据下载

使用 [UCI ADULT](https://archive.ics.uci.edu/ml/datasets/adult) 数据集，
下载数据集后，解压得到训练数据adult.data 和测试数据 adult.test，
将adult.data改名为adult.train.csv，将adult.test改名为adult.test.csv，放置到项目目录下

## 数据预处理

运行data_process.py，进行数据预处理
```
python data_process.py
```

## 单机训练

运行train_single.py，进行单机训练
```
python train_single.py
```

## 单机纵向联邦学习

假设有3个参与方，分别为A,B,C

打开三个终端，分别输入命令：

```
RENDEZVOUS=file:///tmp/vfl WORLD_SIZE=3 RANK=0 python train_multi.py
```

```
RENDEZVOUS=file:///tmp/vfl WORLD_SIZE=3 RANK=1 python train_multi.py
```

```
RENDEZVOUS=file:///tmp/vfl WORLD_SIZE=3 RANK=2 python train_multi.py
```

即可开始单机纵向联邦学习

## 多机纵向联邦学习

假设有3个参与方，分别为A,B,C

A的IP为192.168.1.100，B的IP为192.168.1.101，C的IP为192.168.1.102，A,B,C可以互相访问
将项目文件分别复制到3台机器下，
将dataset/a下的数据复制到A中，dataset/b下的数据复制到B中，dataset/c下的文件复制到C中

在A,B,C三台机器的终端中，分别输入命令

A
```
RENDEZVOUS=tcp://192.168.1.100:2345 WORLD_SIZE=3 RANK=0 python train_multi.py
```

B
```
RENDEZVOUS=tcp://192.168.1.100:2345 WORLD_SIZE=3 RANK=1 python train_multi.py
```

C
```
RENDEZVOUS=tcp://192.168.1.100:2345 WORLD_SIZE=3 RANK=2 python train_multi.py
```

即可开始单机纵向联邦学习

## 训练结果对比

单机 vs 纵向联邦学习

| epoch | 单机auc | 纵向联邦学习auc |
| --- | --- | --- |
| 1 | 0.635 | 0.626 |
| 2 | 0.766 | 0.755 |
| 3 | 0.813 | 0.809 |
| 4 | 0.831 | 0.831 |
| 5 | 0.841 | 0.843 |
| 6 | 0.847 | 0.850 |
| 7 | 0.852 | 0.856 |
| 8 | 0.856 | 0.859 |
| 9 | 0.859 | 0.863 |
| 10 | 0.862 | 0.866 |
| 11 | 0.864 | 0.868 |
| 12 | 0.867 | 0.871 |
| 13 |  0.869 | 0.873 |
| 14 | 0.871 | 0.876 |
| 15 | 0.874 | 0.878 |
| 16 | 0.876 | 0.880 |
| 17 | 0.878 | 0.882 |
| 18 | 0.879 | 0.884 |
| 19 | 0.881 | 0.885 |
| 20 | 0.883 | 0.887 |

