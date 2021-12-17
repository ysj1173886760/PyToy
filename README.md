# PyToy

#### Still Under Construction

参考[MatrixSlow](https://github.com/zackchen/MatrixSlow)基于numpy和cupy实现的简易的深度学习框架

实现了常见的算子，优化器，基础的损失函数等

kernel：卷积，BatchNormalization，ReLU，Dropout，池化

optimizer：AdaGrad, RMSProp, Adam

loss: CrossEntropyWithSoftmax, L2Loss

基于ParameterServer的分布式训练

通过算子封装常用的layer

在test中有框架具体的使用用例

绘制计算图示例

![20211217144418](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20211217144418.png)

受到这个[项目](https://github.com/danistefanovic/build-your-own-x)的启发，之后会我会根据这个框架写一个教程，教大家实现一个自己的深度学习框架

当然还是推荐大家去看一下MatrixSlow原作者的书，这里实现的PyToy也是相当于对原框架的一个增强

### Contribution Guidance

项目中有一个todo_list，我会在后面的时间慢慢实现todo_list的内容，也欢迎大家与我联系一起开发

*****

An simple deeplearning framework pytoy inspired by MatrixSlow(see the link above)

I've implemented common operators, optimizers, loss function. Check the brief description above

Also i've implemented distributed training based on ParameterServer architecture.

Code Architecture:

pytoy/core: kernel of this framework. Basically it's the abstaction classes and the core part of compute graph

pytoy/layer: encapsulation of the common operators

pytoy/distribute: distributed training framework based on gRPC and protobuf

pytoy/ops: main part of this framework, operators and loss functions are in there

pytoy/optimizer: basic optimizers used in deep learning

pytoy/trainer: trainer, mostly it's the optimization for effiency training and the abstraction of distributed training

pytoy/utils: utils used in this framework

test: examples which shows how you can utilize this framework. I tested it to training cifar