'''
该架构的缺点
模型很简单易懂，但是没有区分训练集和测试集，模型不具有普遍性
同时当前的疑惑点在于如何将loss的取值和optimizer联系起来

'''
# imports
import torch
import torch.nn as nn
import torch.optim as optim # SGD Adam
import torch.nn.functional as F # 不需要参数的地方
from torch.utils.data import DataLoader
import torchvision.datasets as  datasets
import torchvision.transforms as transforms


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self,input_size,num_classes): # 28*28
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 784 # 28*28 因为后续的操作将其进行展平，是全连接层因此只能进行展平操作
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epoches = 1

# Load Data
train_dataset = datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset = datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

# Initialize network
model = NN(input_size=input_size,num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # 构建损失函数
optimizer = optim.Adam(model.parameters(),lr=learning_rate) # 构建优化器

# Train Network
# epoch意味着可以模型可以看到所有的数据
for epoch in range(num_epoches):
    for batch_idx,(data,target) in enumerate(train_loader): # train_loader会自动生成数据和label
        # 需要将其都存储到gpu上
        data = data.to(device=device)
        target = target.to(device=device)
        
        # Get to correct shape
        data = data.reshape(data.shape[0],-1)
        
        # forward
        # NN类中的forward方法定义了模型前向传播的逻辑，当调用model(input_data)时
        # Pytroch会自动调用forward方法，并将input_data作为参数传入，进行数据处理并
        # 返回处理后的结果
        scores = model(data)
        loss = criterion(scores,target) # 计算损失
        
        # backward
        # 针对每个batch将所有的梯度置为0
        optimizer.zero_grad()
        loss.backward() # 求出更新weights.grad的值
        
        # gradient descent or adam step
        optimizer.step() # 根据新的weights.grad值更新迭代weights值
        
# Check accuracy n training & test to see how good our model

def check_accuracy(loader,model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0 
    num_samples = 0
    
    # 不希望模型在测试的时候学习
    model.eval()
    # 不需要计算梯度
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0],-1)
            
            # 此时在确定正确率的时候不需要进行前向传播和反向传播了
            # 64 X 10
            scores = model(x)
            _, prediction = scores.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)
        
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    # 结束评估使模型正常运行
    model.train()


check_accuracy(train_loader,model)
check_accuracy(test_loader,model)