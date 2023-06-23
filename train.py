import torch.cuda

from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from net import MyModel

writer = SummaryWriter(log_dir='logs')

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 训练数据集
train_data_set = datasets.CIFAR10("./dataset", train=True, transform=transform, download=True)




# 测试数据集
test_data_set = datasets.CIFAR10("./dataset", train=False, transform=transform, download=True)

train_data_size = len(train_data_set)
test_data_size = len(test_data_set)

print("训练集：{},验证集:{}".format(train_data_size, test_data_size))
# 加载数据集
train_data_loader = DataLoader(train_data_set, batch_size=64,shuffle=True)
test_data_loader = DataLoader(test_data_set, batch_size=64)

# 网络定义
myModel = MyModel()

# 是否使用gpu
use_gpu = torch.cuda.is_available()
if (use_gpu):
    print("gpu可用")
    myModel = myModel.cuda()

# 训练轮数
epochs = 300
# 损失函数
lossFn = nn.CrossEntropyLoss()
# 优化器
optimizer = SGD(myModel.parameters(), lr=0.01)
for epoch in range(epochs):
    print("训练轮数{}/{}".format(epoch + 1, epochs))

    # 损失变量
    train_total_loss = 0.0
    test_total_loss = 0.0
    # 精度
    train_total_acc = 0.0
    test_total_acc = 0.0

    # 训练开始
    for data in train_data_loader:
        inputs, labels = data
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()  # 清空梯度
        outputs = myModel(inputs)  # 前向运行

        loss = lossFn(outputs, labels)  # 计算损失
        # 计算精度
        _, pred = torch.max(outputs, 1)  # 得到预测值得索引
        acc = torch.sum(pred == labels).item()  # 和目标进行对比 得到精度
        loss.backward()  ## 误差反向传播
        optimizer.step()  # 参数更新

        train_total_loss += loss.item()
        train_total_acc += acc

    # 测试
    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = myModel(inputs)  # 前向运行
            loss = lossFn(outputs, labels)  # 计算损失
            # 计算精度
            _, pred = torch.max(outputs, 1)  # 得到预测值得索引
            acc = torch.sum(pred == labels).item()  # 和目标进行对比 得到精度

            test_total_loss += loss.item()
            test_total_acc += acc

    print("train loss:{},acc:{}. test loss:{},acc:{}".format(train_total_loss, train_total_acc / train_data_size,
                                                             test_total_loss, test_total_acc / test_data_size))
    writer.add_scalar('Loss/train', train_total_loss, epoch)
    writer.add_scalar('Loss/test', test_total_loss, epoch)
    writer.add_scalar('acc/train', train_total_acc / train_data_size, epoch)
    writer.add_scalar('acc/test', test_total_acc / test_data_size, epoch)
    if((epoch + 1)% 50 == 0 ):
        torch.save(myModel, "model/model_{}.pth".format(epoch+1))
# tensorboard --logdir=logs --port=6007