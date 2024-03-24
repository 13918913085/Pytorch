import torch

'''
#  Tensor Reshaping
基本不会对原始变量进行操作，都需要手动赋予新的变量
- x.view()
- x.reshape()
- torch.cat((x1,x2),dim=0)
- x.permute()
- x.unsqueeze()
- x.squeeze()
'''

x = torch.arange(9)

x_3x3 = x.view(3,3)
print(x_3x3)
x_3x3 = x.reshape(3,3)

y = x_3x3.t()
print(y.contiguous().view(9))

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
# 拼接的话是数值相加
print(torch.cat((x1,x2),dim=0).shape)
print(torch.cat((x1,x2),dim=1).shape)

# 将x1 进行展平处理
z = x1.view(-1)
print(z.shape)

batch = 64
x = torch.rand((batch,2,5))

# 保存第一个维度，但是将后面两个维度进行展平处理
z  = x.view(batch,-1)
print(z.shape)

# 利用indice的坐标交换原始的坐标
# transpose 是permute的一个特殊情况
z = x.permute(0,2,1)

x = torch.arange(10) # [10]
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)
print(x.shape)
z = x.squeeze(1)
print(z.shape)