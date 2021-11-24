0.0.1:

![20211124175319](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20211124175319.png)

网络结构

![20211124130322](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20211124130322.png)

收敛非常迅速，大概在10个epoch到20个epoch左右就可以达到50%到60%的正确率，但是随着epoch提高，loss和准确率会收敛到一个值

用了bn以后确实训练效果好了很多，收敛速度也会加快，准确率提升的十分明显。

但是这个正确率却没有达到我的预期，猜测可能与我只用了第一个databatch训练有关

也有可能与网络结构有关

0.0.2:

![20211124175253](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20211124175253.png)

网络结构没变，增大了测试数据和训练数据的数据量，效果还是基本没变化

大概率是网络结构的原因了

0.0.3:

![20211124191831](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20211124191831.png)

这次我添加了adam和多一个全连接层，效果反而变差了。不太清楚是adam的原因还是全连接层的原因

貌似只是因为我训练的epoch太少了

0.0.4

![20211124214430](https://picsheep.oss-cn-beijing.aliyuncs.com/pic/20211124214430.png)

这个是原本的结构加上adam，没有多的全连接层

感觉还有上升的余地，但是就先50个epoch吧

感觉下一步应该从数据入手一下，看看能不能增强一下数据。现在这个情况有上有下的就感觉是训练的不好

也可能是特征太少了。