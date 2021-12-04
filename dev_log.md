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

而且速度基本不变

极限情况下可以显式调用cupy释放显存，但是速度会从2.0x 掉到1.7x

显存不会超过1000,平均情况下是500左右