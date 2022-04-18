11.30

![20211130095125](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20211130095125.png)

第一步成功，成功拟合了一个3元的线性方程

![20211130202224](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20211130202224.png)

mnist训练成功

美妙的计算图

12.1

![20211201175609](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20211201175609.png)

实现了cifar，但是莫名其妙的变快了，我自己都不知道为什么

按理说这个写法应该是会慢一些才对

![20211201202833](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20211201202833.png)

修复画图后的计算图

12.2

实现保存加载模型

12.3

修正conv， 实现residual block

12.4

存储优化

优化前cifar

![20211204132530](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20211204132530.png)

优化后大概是2000,优化了600mb

应该还可以继续

进一步优化达到了1600mb

速度会有一定的下降，但是会随着显存的使用变得不是那么明显

小模型还是用普通的训练方法就好

极限情况下可以显式调用cupy释放显存，但是速度会从2.0x 掉到1.7x

显存不会超过1000,平均情况下是500左右

折中情况下可以实现一个background vacumming

background vacuumming 效果不错

12.5

实现了分布式训练 PS模式

修正了BN节点的不同步的情况

同步模式下的PS训练瓶颈还是在同步节点上

下一步要实现异步的PS模式的训练

以及BN中方差的估计方法要进行修正

接下来也要考虑不同机器不同batch大小的情况