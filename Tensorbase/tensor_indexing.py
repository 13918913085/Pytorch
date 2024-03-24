import torch

'''
Tensor Indexing
'''

batch_size = 10 
feature = 25
x = torch.rand((batch_size,feature))
print(x[0].shape) # x[0,:] 25

print(x[:,0].shape) # 10

print(x[2,0:10].shape)

x[0,0] = 100

# Fancy

x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices])

x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
# 此时返回(1,4)和(0,0)两个位置上的数据
print(x[rows,cols].shape)


# More advance indexing

x = torch.arange(10)
# (x < 2)此时返回一个与x大小相同的tensor向量，用False或者True来确定
# x[(x < 2)]则会输出那些index为True的数值
print(x[(x < 2) | (x>8)])
# remainder 表示的是余数的情况
print(x[x.remainder(2) == 0])


# Userful operations 

# 当x>5的时候，取值不变，否则取值变为两倍
print(torch.where(x>5,x,x*2))
# 类似于字典，将所有的取值变成单一化
print(torch.tensor([0,0,1,2,2,3,4]).unique())
# 确定x的维度是多少
print(x.ndimension())
# 计算x中含有的数据是多少
print(x.numel())
