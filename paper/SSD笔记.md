### SSD

> Single Shot MultiBox Detector

[论文地址](https://arxiv.org/abs/1512.02325)


#### 优点和缺点

1. 识别精度高。
2. 速度快，能够达到实时效果，是端到端算法。

3. 对大目标检测效果好，对小目标检测效果差。
4. 对不同宽高比的物体检测效果好，使用了不同aspect ratios的default boxes。


#### 训练输入

1. 图像
2. 标签
3. 边界框(Bounding Box)


#### 其他方案

**RCNN系列**
1. 生成一些推荐的bounding boxes
2. 在这些bounding boxes中提取特征
3. 通过分类器来分类


#### prediction阶段

* 计算出每一个default box中的物体属于每个类别的可能性。
* 对这些default box的shape进行微调，以使其符合物体的外接矩形。
* 为了处理相同物体的不同尺寸的情况，结合了不同分辨率的feature maps的predictions。


#### 主要贡献

* 


#### 模型

> SSD 是一个前向传播的CNN网络，产生一系列固定大小的bounding boxes，
以及每一个box中包含物体实例的得分。然后进行一个非极大值抑制得到最终的predictions。

**基础网络(base network)**

用图像分类的标准架构


**辅助网络**

* Multi-scale feature maps for detection(用于检测的多尺度特征图)

在基础网络之后，添加了额外的卷积层，这些卷积层的大小是逐层递减的，可以在`多尺度`下进行predictions。

* Convolutional predictors for detection(用于检测的卷积层预测器)

没看懂

* Default boxes and aspect ratios(默认boxes和宽高比)

每一个box相对于与其对应的feature map cell的位置是固定的，在每一个feature map cell中，
要预测得到predict得到的box与default box之间的`offsets`,以及每一个box中包含物体的`score`。

对于每一个位置上的k个boxes中的每一个，需要计算出C个类别的`score`，
和这个box相对于default box的`offset`。于是，在feature map中的每一个cell上，
就需要有`(c+4)*k`个`filters`,对于一张`m*n`的feature map,会产生`(c+4)*k*m*n`个输出结果。


#### 训练

> SSD训练图像中的groundtruth需要赋予到那些固定输出的boxes上，
SSD输出一系列定义好的固定大小的bounding boxes。


* Matching strategy


* Training objective


* Choosing scales and aspect ratios for default boxes

大部分 CNN 网络在越深的层，feature map 的尺寸（size）会越来越小。
这样做不仅减少了对计算与内存的需求，而且在最后提取的feature map会有某种程度上的平移与尺度不变性。

为了处理不同尺寸的物体，一些模型将图像转换成不同的尺度，将这些图像独立的通过CNN网络处理，
再将这些不同尺度的图像结果进行综合。

如果使用同一个网络中的、不同层上的 feature maps，也可以达到相同的效果，
同时在所有物体尺度中共享参数。

因此，SSD同时使用`lower feature maps`、`upper feature maps`来预测`predict detections`。


* Hard negative mining

在生成一系列的 predictions 之后，会产生很多个符合 ground truth box 的 predictions boxes，
但同时，不符合 ground truth boxes 也很多，而且这个 negative boxes，远多于 positive boxes。
这会造成 negative boxes、positive boxes 之间的不均衡,使得训练时难以收敛。

因此，本文先将每一个物体位置上对应 predictions（default boxes）是 negative 的 boxes 
进行排序，按照 default boxes 的 confidence 的大小。
选择最高的几个，保证最后 negatives、positives 的比例在 3:1。


* Data augmentation

对于每一张训练图像，随机进行如下几种选择：
   1. 使用原始的图像
   2. 采样一个patch,与物体之间最小的jaccard overlap为0.1,0.3,0.5,0.7与0.9。
   3. 随机采样一个patch

每一个采样的patch被resize到固定的大小，并且以0.5的概率随机的水平翻转。



#### 实验分析

* 数据增强对结果的提升非常明显

* 使用更多的(lower layer feature maps)特征图对结果提升很大

* 使用更多的default boxes，结果也越好

* Atrous使SSD的结果又好又快



#### Reference

* [【深度学习：目标检测】RCNN学习笔记(10)：SSD:Single Shot MultiBox Detector](http://blog.csdn.net/smf0504/article/details/52745070)

