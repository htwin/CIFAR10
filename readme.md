# 代码使用指南

这个是一个入门的一个代码，模型是手写的，效果肯定是比不了那些很成熟的模型，比如yolo系列。

但是自己拿来学习 从头写一下也是不错的，毕竟以后手写的机会很少了。





## 视频教程

代码是我全部手敲完成，全程实录，视频已上传到b站。

[点击查看视频](https://www.bilibili.com/video/BV1dT411z7mm/)

## 测试

直接运行test.py 文件，进行测试，在model文件夹下保存着预训练50轮的模型(model_50.pth)

```
python test.py
```



## 进行训练

直接运行 train.py ，本地没有cifar10数据集的话，会默认进行下载。

```
python train.py
```





## 训练可视化

损失函数 和 准确率 通过 tensorboard 可视化

```
# 日志文件存放在logs 执行以下命令 即可访问 
tensorboard --logdir=logs --port=6007
```









