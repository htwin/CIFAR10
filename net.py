import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, 5,1,2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.liner1 = nn.Linear(1024,64)
        self.liner2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)


    def forward(self,x):

        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.liner1(x)

        x = self.liner2(x)
        x = self.softmax(x)

        return x;

#
# x = torch.randn(1,3,32,32)
# myModel = MyModel()
# out = myModel(x)
# print(out)