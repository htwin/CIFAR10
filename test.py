import torch
import os
import  cv2
import torchvision
from PIL import Image
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

use_gpu = torch.cuda.is_available()
model = torch.load("model/model_50.pth", map_location=torch.device('cuda' if use_gpu else 'cpu'))
model.eval()
# 指定文件夹路径
folder_path = './testimages'

# 获取文件夹中所有文件
files = os.listdir(folder_path)

# 过滤出所有图片文件
image_files = [os.path.join(folder_path, f) for f in files if f.endswith('.jpg') or f.endswith('.png')]

# 输出所有图片文件路径
for img in image_files:
    image =cv2.imread(img)
    cv2.imshow('image', image)
    image = cv2.resize(image,(32,32))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    image = transform(image)
    image = torch.reshape(image,(1,3,32,32))
    image = image.to('cuda' if use_gpu else 'cpu')
    output = model(image)
    value,index = torch.max(output.data,1)
    pre_val = classes[index]
    print(output)
    print("预测值：{}，预测索引:{},预测结果：{}".format(value.item(),index.item(),pre_val))


    # 等待用户按下键盘
    cv2.waitKey(0)
