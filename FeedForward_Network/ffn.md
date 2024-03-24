## loss.backward()和optimizer.zero_grad()以及optimizer.step()
(Pytorch 疑案之：优化器和损失函数是如何关联起来的？)[https://blog.csdn.net/rizero/article/details/104185046]

## super(NN,self).__init__()
这是对继承自父类的属性进行初始化，而且是用父类的初始化方法来初始化继承的属性。也就是说，子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化

## model.train() 和 model.eval() 作用
- model.train()和model.eval()是PyTorch中用于将模型设置为训练模式和评估模式的方法
- 在深度学习中，我们通常使用训练数据对模型进行训练，并使用测试数据对模型进行评估。在训练和测试阶段，模型的行为可能会有所不同，特别是模型包含Batch Normalization和Dropout等层时，这些层在训练和测试阶段的行为可能不同。因此，我们需要根据不同的模式设置模型的行为，以保证模型的正常工作
- model.train()方法用于将模型设置为训练模式，主要作用时使Batch Normalization和Dropout等层正常工作。在训练模式下Batch Normalization层会使用当前batch的统计信息（均值和方差）进行归一化，以加速训练过程；而Dropout层会随机丢弃一部分神经元，以防止过拟合。
- model.eval() 方法用于将模型设置为评估模式，主要作用是使 Batch Normalization 和 Dropout 等层正常评估。在评估模式下，Batch Normalization 层会使用所有训练数据的统计信息（均值和方差）进行归一化，以保证模型的稳定性；而 Dropout 层会保留所有神经元，以提高模型的准确性。因此，在评估模式下，我们需要保证 Batch Normalization 和 Dropout 等层的评估工作，以使模型能够得到更好的评估效果
