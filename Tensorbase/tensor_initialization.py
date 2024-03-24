import torch 

'''
Initializing Tensor

'''

device = "cuda" if torch.cuda.is_available() else "cpu"
# require_grad 是用来判断是否需要用来计算梯度回传的情况
my_tensor = torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32,
                         device=device,requires_grad=True)

# 查看tensor本身具有的属性
# print(my_tensor)
# print(my_tensor.dtype)
# print(my_tensor.device)
# print(my_tensor.shape)
# print(my_tensor.requires_grad)


'''
other common initialization methods
'''

# 此时的数据是随机数的情况，不一定是全零
x = torch.empty(size=(3,3))  
x = torch.zeros((3,3))
# 在0~1之间，均匀分布生成的张量
x = torch.rand((3,3))
# 生成均值为0，方差为1的标准正态分布 
x = torch.randn((3,3))
x = torch.ones((3,3))
# Indentity matirx
x = torch.eye(5,5) 
x = torch.arange(start=0,end=5,step=1)
# ten values bettween start and end(包含start和end)
x = torch.linspace(start=0.1,end=1,steps=10)
# 生成均值为0方差为1的一行五列的正态分布的数据
x = torch.empty(size=(1,5)).normal_(mean=0,std=1)
# 生成0~1之间均匀分布的数据
x = torch.empty(size=(1,5)).uniform_(0,1)
# 在此种情况下类似于eye 数组
x = torch.diag(torch.ones(3))


'''
How to initialize and convert tensors to other types
                                    (int,float,double)
'''

tensor = torch.arange(4)
print(tensor.bool()) # boolean True/False
print(tensor.short()) # int 16
print(tensor.long()) # int 64 (Important)
print(tensor.half()) # float 16
print(tensor.float()) # float 32(Important)
print(tensor.double()) # float 64

'''
Array to Tensor conversion and vice-versa
'''
import numpy as np

np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()