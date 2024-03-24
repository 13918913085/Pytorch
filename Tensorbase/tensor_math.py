import torch

x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

'''
Basic operation
'''
# Addition
z1 = torch.empty(3)
torch.add(x,y,out=z1)
z2 = torch.add(x,y)
z = x + y 

# Subtraction
z = x - y

# Division
z = torch.true_divide(x,y)

# inplace operations
# 在pytorch中改变一个tensor值的时候，不经过复制操作
# 而是直接在原来的内存上改变它的值
t = torch.zeros(3)
t.add_(x)
t += x

# Exponentiation
z = x.pow(2)
z = x ** 2

# Simple comparison

z = x > 0
z = x < 0

# Matirx Multiplication 
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

# matrix exponentiation

matrix_exp = torch.rand(5,5)
matrix_exp.matrix_power(3)

# element wise mult
z = x * y
print(z)

# dot product
# 此时是相乘之后还需要进行相加操作
z = torch.dot(x,y)
print(z)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch,n,m))
tensor2 = torch.rand((batch,m,p))
out_bmm = torch.bmm(tensor1,tensor2) # (batch,n,p)

# Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

# 维度为1的数据会自动匹配其他数据
z = x1 - x2
z = x1 ** x2 

# Other useful tensor operation
sum_x = torch.sum(x,dim=0)
# 最大值返回当前的数值，和数值所处的坐标
values,indices = torch.max(x,dim=0)
values,indices = torch.min(x,dim=0)
abs_x = torch.abs(x)
# 此时和max操作类似，只是只返回x的索引，不返回最大值
z = torch.argmax(x,dim=0)
z =torch.argmin(x,dim=0)
# mean操作需要输入的数据为浮点数的形式
mean_x = torch.mean(x.float(),dim=0)
z = torch.eq(x,y)
# 返回排序好的数据，以及数据修改的列表形式
sorted_y,indices = torch.sort(y,dim=0,descending=False)

# 将所有小于0的数据置为0，大于10数据置为10
z = torch.clamp(x,min=0,max=10)

x = torch.tensor([1,0,1,1,1],dtype=torch.bool)
# 只要x中的数据中有一个是True z的取值为True
z = torch.any(x)
z = torch.all(x)
print(z)



