import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim  as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义一个简单的全连接层结构
class NN(nn.Module):
    # __init__要有下划线，同时super后面有括号
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义所使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义超参数
input_size = 784
num_classes = 10
batch_size = 64
lr = 0.001
epoches = 1

# 定义数据集
# 是在读取数据集的时候就确定了数据集的格式
train_dataset = datasets.MNIST(root = 'dataset/',train = True,download = False,transform = transforms.ToTensor())
train_dataloader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
test_dataset = datasets.MNIST(root = 'dataset/',train = False,download = False,transform = transforms.ToTensor())
test_dataloader = DataLoader(dataset = test_dataset,batch_size = batch_size,shuffle = True)


# 定义模型 别忘了将其映射到cuda上
model = NN(input_size=input_size,num_classes=num_classes).to(device)

# 定义目标函数
criterion = nn.CrossEntropyLoss()
# 需要在构建优化器的时候确定学习率
optimizer = optim.Adam(model.parameters(),lr = lr)

# 开始来进行计算
for epoch in range(epoches):
    for idx,(data,label) in enumerate(train_dataloader):
        data = data.to(device) # 需要显式的定义，否则不对data本身进行迁移
        label = label.to(device)
        data = data.reshape(data.shape[0],-1)
        
        # 开始进行前向计算
        pred = model(data)
        loss = criterion(pred,label)
        
        # 开始反向传播
        optimizer.zero_grad() # 优化器对其进行梯度归零
        
        loss.backward()
        optimizer.step()
def check_accuracy(loader,model):
    num_correct = 0
    num_total = 0
    model.eval()
    
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        # reshape的方法也需要显式的赋值
        x = x.reshape(x.shape[0],-1)
        score = model(x)
        _ , pred = score.max(dim=1)
        num_correct += (pred==y).sum()
        # 此处应该也可以用shape
        num_total += pred.shape[0]
    print(f'{num_correct}/{num_total}')
    # 回归原来的参数
    model.train()

    
check_accuracy(train_dataloader,model)