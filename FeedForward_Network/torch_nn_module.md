(Pytorch中文网址)[https://ptorch.com/docs/1/utils-data]

## torch.nn
    torch.nn是Pytorch中用于构建神经网络的模块，它提供了一组类和函数，用于定义、训练和评估神经网络模型。
    torch.nn 模块的核心是nn.Module类，它是所有神经网络模型的基类，在Containers中，通过继承nn.Module类，可以创建自己的神经网络模型，并定义模型的结构和操作
    常用的的一些类和函数
    - nn.Linear: 线性层，用于定义全连接层
    - nn.Conv2d: 二维卷积，用于处理图像数据
    - nn.ReLU: ReLU激活函数
    - nn.Sigmoid: Sigmoid 激活函数
    - nn.Dropout: Dropout 层，用于正则化或防止过拟合
    - nn.CrossEntropyLoss: 交叉熵损失函数，通常用于多类别分类问题
    - nn.MSELoss: 均方误差损失函数，通常用于回归问题
    - nn.Sequential: 顺序容器，用于按顺序组合多个层

    使用torch.nn模块，您可以创建自定义的神经网络模型，并使用Pytorch提供的优化器（如torch.optim）和损失函数来训练和优化模型

## torch.nn.functional
    - torch.nn.functional.threshold(input, threshold, value, inplace=False)
    - torch.nn.functional.relu(input, inplace=False)
    - torch.nn.functional.relu6(input, inplace=False)
    - torch.nn.functional.elu(input, alpha=1.0, inplace=False)
    - torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False)
    - torch.nn.functional.prelu(input, weight)
    - torch.nn.functional.rrelu(input, lower=0.125, upper=0.3333333333333333, training=False, - inplace=False)
    - torch.nn.functional.logsigmoid(input)
    - torch.nn.functional.softmax(input)
    - torch.nn.functional.log_softmax(input)
    - torch.nn.functional.tanh(input)
    - torch.nn.functional.sigmoid(input)

## torch.nn.functional 中函数和torch.nn中的函数的区别
    torch.nn.functional中的函数和torch.nn中的函数都提供了常用的神经网络操作，包括激活函数、损失函数、池化操作等。它们的主要区别如下：
    - 函数形式 vs 类形式
    torch.nn.functional中的函数是以函数形式存在的，而torch.nn中的函数是以类形式存在的。torch.nn.functional中的函数是纯函数，没有与之相关联的可学习参数。而torch.nn中的函数是torch.nn.Module的子类，可以包含可学习参数，并且可以在模型中作为子模块使用。
    - 参数传递方式
    torch.nn.functional中的函数时直接传递张量作为参数的，而torch.nn中的函数需要实例化后，将张量作为实例的调用参数
    - 状态管理
    由于torch.nn.functional中的函数是纯函数，没有与之相关联的参数或状态，因此无法直接管理和访问函数的内部状态。而torch.nn中的函数是torch.nn.Module的子类，可以管理和访问模块的内部参数和状态

    如果希望激活函数只需要在前向传播中使用，那么使用torch.nn.functional中的激活函数更加简洁。如果希望将激活函数作为模型的一部分，并与其他模块一起使用，则使用torch.nn中的激活函数更加方便
    
## torch.optim
    torch.optim 是实现各种优化算法的包，最常用的方法都以支持，接口很常规，以后也可以很容易地集成更复杂的算法
    - 要构造一个Optimizer，必须给他一个包含参数（必须都是Variable对象）进行优化。然后指定optimizer的参数选项，比如学习率，权重衰减等
    optimizer = optim.SGD(model.parameters(),lr = 0.01, momentum = 0.9)
    optimizier = optim.Adam([var1,var2],lr=0.0001)
    - Optimizer也支持为每个参数单独设置选项。若想这么做，不要直接传入Variable的iterable，而是传入dict的iterable。每一个dict都分别定 义了一组参数，并且包含一个param键，这个键对应参数的列表。其他的键应该optimizer所接受的其他参数的关键字相匹配，并且会被用于对这组参数的优化。
    optim.SGD([
        {'params':model.base.parameters()},
        {'params':model.classifier.parameters(),'lr':1e-3}
        ],lr=1e-2,momentum=0.9)
        这意味着model.base参数将使用默认的学习速率1e-2，model.classifier参数将使用学习速率1e-3，并且0.9的momentum将会被用于所有的参数。
    - 所有的optimizer都会实现step()更新参数的方法
    optimizer.step()
    这是大多数optimizer所支持的简化版本，一旦梯度被如backward()之类的函数计算好后，就可以调用该函数
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

## torch.utils.data.DataLoader
    (Pytorch中的torch.utils.data模块)[https://blog.csdn.net/hanmo22357/article/details/129710074]

    torch.utils.data.DataLoader 是一个Pytorch中用于批量加载数据的工具类。它可以将自定义数据集（如torch.utils.data.Dataset或torch.utils.data.TensorDataset）转换为一个可迭代对象，并支持多线程和批量加载等功能
    torch.utils.data.DataLoader类的构造函数又许多可用参数，以下是一些主要的参数
    - dataset: 必须参数，指定要加载的数据集
    - batch_size: 每个批次包含的样本数，默认为1
    - shuffle: 是否对数据进行随机化处理, 默认为False
    - sampler: 指定从数据集中采样样本的策略，若指定此参数，则shuffle参数无效
    - batch_sampler：指定从数据集中采样批次的策略，若指定此参数，则 batch_size 和 shuffle 参数无效。
    - num_workers：用于数据加载的子进程数，默认为 0（单线程）。对于Window系统这个参数只能是0。
    - collate_fn：用于对样本进行自定义处理的函数，例如对不同长度的样本进行填充等。一般不使用这个参数。
    - pin_memory：是否将数据加载到固定内存中，默认为 False。设置为True可以提高数据加载速度，但是也会占用更多的内存，并且只对于GPU计算有用。建议在使用GPU进行计算时都将该参数设置为True。
    - drop_last：如果数据集大小不能被批次大小整除，是否将最后一个小于批次大小的批次丢弃，默认为 False。
    - timeout：数据加载超时时间，默认为 0，表示无限等待。

## torchvision.transforms
    torchvision.transforms是pytorch中的图像预处理包，包含了一些常用的图像变换，主要实现对数据集的预处理、数据增强、数据转换成tensor等一些列操作，一般使用Compose把多个步骤整合到一起
    ```
    transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    ```
    - transforms.Resize()
    将输入图片大小调整为指定大小
    - transforms.ToTensor()
    把PIL图像或[0,255]范围内的numpy.ndarray（形状H x W x C）转换为torch.FloatTensor,张量形状(C x H x W),范围在[0.0,1.0]中
    - transforms.Normalize(mean,std)
    用平均值和标准差标准化输入图片，给定n个通道的平均值(M1,...,Mn)和标准差(S1,...,Sn),这一变换会在输入图片的每一个通道上进行标准化,即input[channel] = (input[channel] - mean[channel]) / std[channel]。
        mean：序列，包含各通道的平均值
        std：序列，包含各通道的标准差

## torchvision.datasets
    torchvision.datasets中包含了以下数据集
    - MNIST
    - COCO（用于图像标注和目标检测）(Captioning and Detection)
    - LSUN Classification
    - ImageFolder
    - Imagenet-12
    - CIFAR10 and CIFAR100
    - STL10

    eg:
    dset.MNIST(root, train=True, transform=None, target_transform=None, download=False)
    参数说明： 
    - root : processed/training.pt 和 processed/test.pt 的主目录 
    - train : True = 训练集, False = 测试集 
    - download : True = 从互联网上下载数据集，并把数据集放在root目录下. 如果数据集之前下载过，将处理过的数据（minist.py中有相关函数）放在processed文件夹下
