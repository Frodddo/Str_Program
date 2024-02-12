# 项目食用指南：

1.大家每次对README.md文件、模型文件等进行更新或修改操作后，需要在update.md中登记，格式在update.md中有呈现

2.在学习完某个模型时，将学习到的内容放在record.md文件中，可以进行详细步骤阐述，格式不限，并将代码放在Github Programs文件夹中

3.祝我们开展项目顺利

**版本号：**

**1.1 by-dxo 2024/02/12（大家每次修改完记得也在这里写下）**

**1.2 by-**

# 1 前期准备

## 软件下载（选择性安装）

1.Softwares文件夹 Win v1.8.10中为Typora软件及激活码，按照视频或图文步骤来激活Typora，github项目中.md后缀文件可用Typora打开

Typora快速入门[用markdown写下你的第一个md文档 - 简书 (jianshu.com)](https://www.jianshu.com/p/de9c98bba332)

2.Softwares文件夹 ZhiyunTrans中为知云文献翻译安装包，墙裂推荐，使用方便

3.Softwares文件夹 IDM中为Internet Download Manager软件 可以快速下载网页视频与Github项目

4.VPN（便宜挺好用）：[一元机场 (xn--4gq62f52gdss.com)](https://xn--4gq62f52gdss.com/#/login)

## 视频教程

1.Github快速入门（会的直接跳过）

[『教程』一看就懂！Github基础教程_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1hS4y1S7wL/?spm_id_from=333.337.search-card.all.click&vd_source=9d0873d2d65bb6fe93af166e55a35858)

2.Python快速入门

[https://www.bilibili.com/video/BV1qW4y1a7fU/?spm_id_from=333.337.search-card.all.click](https://www.bilibili.com/video/BV1qW4y1a7fU?p=4&vd_source=9d0873d2d65bb6fe93af166e55a35858)

3.斯坦福卷积神经网络快速入门（我上学期做计算思维实训学的 感觉讲得很好很详细 卷积神经网络是这个项目基础的基础！）

https://www.bilibili.com/video/BV1nJ411z7fe/?spm_id_from=333.337.search-card.all.click

另外找的：（可以二选一 ，感觉这个也不错，其他的也可以）

[神经网络算法精讲！原理+代码复现，半天带你学懂深度学习神经网络算法入门到实战！_AI/人工智能/深度学习/神经网络（CNN/RNN/GAN）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1dk4y1U7PT/?spm_id_from=333.999.0.0&vd_source=9d0873d2d65bb6fe93af166e55a35858)

## 环境配置

需要用到Pytorch或者Tensorflow和其他库，怎么配置环境的我忘记了，大家找到就可以来补充

## 参考资料

1.经典的卷积神经网络

[经典卷积神经网络-CSDN博客](https://blog.csdn.net/weixin_41997940/article/details/123915366?ops_request_misc=%7B%22request%5Fid%22%3A%22170770715416800182158578%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170770715416800182158578&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-123915366-null-null.142^v99^pc_search_result_base5&utm_term=经典卷积神经网络&spm=1018.2226.3001.4187)

2.从R-CNN到Fast R-CNN到Faster R-CNN 到Faster R-CNN到 Mask R-CNN

[深度学习之目标检测R-CNN模型算法流程详解说明（超详细理论篇）_cnn目标检测cvpr-CSDN博客](https://blog.csdn.net/qq_55433305/article/details/131177839?ops_request_misc=%7B%22request%5Fid%22%3A%22170771579516800211528126%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170771579516800211528126&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-131177839-null-null.142^v99^pc_search_result_base5&utm_term=rcnn实现&spm=1018.2226.3001.4187)

[R-CNN史上最全讲解_rcnn-CSDN博客](https://blog.csdn.net/weixin_43702653/article/details/123973629)

[Fast R-CNN讲解_fast rcnn-CSDN博客](https://blog.csdn.net/weixin_43702653/article/details/124002054?spm=1001.2014.3001.5501)

[Faster R-CNN最全讲解_faster rcnn 训练-CSDN博客](https://blog.csdn.net/weixin_43702653/article/details/124045469?spm=1001.2014.3001.5501)

[Mask R-CNN讲解_maskrcnn-CSDN博客](https://blog.csdn.net/weixin_43702653/article/details/124377487?ops_request_misc=%7B%22request%5Fid%22%3A%22170771326016800225574621%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170771326016800225574621&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-124377487-null-null.142^v99^pc_search_result_base5&utm_term=mask rcnn&spm=1018.2226.3001.4187)

3.其他

[自然场景文本检测识别技术综述_transformer-based text detection in the wild-CSDN博客](https://blog.csdn.net/qq_27009517/article/details/103680480?ops_request_misc=&request_id=&biz_id=102&utm_term=自然场景文字识别&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-103680480.142^v99^pc_search_result_base5&spm=1018.2226.3001.4187)

# 2 模型学习与比较（非端到端）

## 2.1 文字检测模块

一步法模型：直接从图像中预测文字区域

两步法模型：首先生成区域提议，然后对这些提议进行分类和精细定位

**One-Stage： YOLO系列**（目标检测），TextBox（水平矩形框），TextBox++（旋转矩形框），PixelLink，CRAFT，**EAST**（允许带角度矩形框或任意四边形覆盖），**SegLink**

**Two-Stage：**R-CNN系列（Fast R-CNN、**Faster R-CNN、Mask R-CNN**（目标检测）），RRPN（Faster R-CNN变种，允许带角度的矩形框覆盖），**CTPN**（水平文字检测，四个自由度）

### 至少需要学习的算法：YOLO EAST SegLink Faster Faster R-CNN Mask R-CNN CTPN

#### 具体学习与代码

**1.CTPN**

**原理：**

[【OCR技术系列之五】自然场景文本检测技术综述（CTPN, SegLink, EAST）_ocr技术综述-CSDN博客](https://blog.csdn.net/fu_shuwu/article/details/84196322?ops_request_misc=&request_id=&biz_id=102&utm_term=自然场景文字识别&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-84196322.142^v99^pc_search_result_base5&spm=1018.2226.3001.4187)（只是原理 代码额外找的）

**详细操作：**

Tensorflow版本代码已补充在Github Programs文件夹中 text-detection-ctpn-master

[【AI实战】手把手教你深度学习文字识别（文字检测篇：基于MSER, CTPN, SegLink, EAST等方法） - 雪饼的个人空间 - OSCHINA - 中文开源技术交流社区](https://my.oschina.net/u/876354/blog/3054322)

**2.SegLink**

**原理：**

**[【OCR技术系列之五】自然场景文本检测技术综述（CTPN, SegLink, EAST）_ocr技术综述-CSDN博客](https://blog.csdn.net/fu_shuwu/article/details/84196322?ops_request_misc=&request_id=&biz_id=102&utm_term=自然场景文字识别&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-84196322.142^v99^pc_search_result_base5&spm=1018.2226.3001.4187)**

**详细操作：**

Tensorflow版本代码已补充在Github Programs文件夹中 seglink-master

[【AI实战】手把手教你深度学习文字识别（文字检测篇：基于MSER, CTPN, SegLink, EAST等方法） - 雪饼的个人空间 - OSCHINA - 中文开源技术交流社区](https://my.oschina.net/u/876354/blog/3054322)

**3.EAST**

**原理：**

**[【OCR技术系列之五】自然场景文本检测技术综述（CTPN, SegLink, EAST）_ocr技术综述-CSDN博客](https://blog.csdn.net/fu_shuwu/article/details/84196322?ops_request_misc=&request_id=&biz_id=102&utm_term=自然场景文字识别&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-84196322.142^v99^pc_search_result_base5&spm=1018.2226.3001.4187)**

**详细操作：**

代码已补充在Github Programs文件夹中 EAST-master

[【AI实战】手把手教你深度学习文字识别（文字检测篇：基于MSER, CTPN, SegLink, EAST等方法） - 雪饼的个人空间 - OSCHINA - 中文开源技术交流社区](https://my.oschina.net/u/876354/blog/3054322)

**4.RCNN**系列

**原理：**

[深度学习之目标检测R-CNN模型算法流程详解说明（超详细理论篇）_cnn目标检测cvpr-CSDN博客](https://blog.csdn.net/qq_55433305/article/details/131177839?ops_request_misc=%7B%22request%5Fid%22%3A%22170771579516800211528126%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170771579516800211528126&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-131177839-null-null.142^v99^pc_search_result_base5&utm_term=rcnn实现&spm=1018.2226.3001.4187)

[R-CNN史上最全讲解_rcnn-CSDN博客](https://blog.csdn.net/weixin_43702653/article/details/123973629)

[Fast R-CNN讲解_fast rcnn-CSDN博客](https://blog.csdn.net/weixin_43702653/article/details/124002054?spm=1001.2014.3001.5501)

[Faster R-CNN最全讲解_faster rcnn 训练-CSDN博客](https://blog.csdn.net/weixin_43702653/article/details/124045469?spm=1001.2014.3001.5501)

[Mask R-CNN讲解_maskrcnn-CSDN博客](https://blog.csdn.net/weixin_43702653/article/details/124377487?ops_request_misc=%7B%22request%5Fid%22%3A%22170771326016800225574621%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170771326016800225574621&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-124377487-null-null.142^v99^pc_search_result_base5&utm_term=mask rcnn&spm=1018.2226.3001.4187)

**详细操作：**

代码已补充在Github Programs文件夹中 py-faster-rcnn-master

[（图像检测1）Py-faster-rcnn-master目录解析_masterrcnn-CSDN博客](https://blog.csdn.net/weixin_44396553/article/details/120917245?ops_request_misc=%7B%22request%5Fid%22%3A%22170771869316800211577144%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=170771869316800211577144&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-11-120917245-null-null.142^v99^pc_search_result_base5&utm_term=faster rcnn crnn&spm=1018.2226.3001.4187)

**5.CRAFT**

**原理：**

待补充

**详细操作：**

pytorch版本代码已补充在Github Programs文件夹中 CRAFT-pytorch-master

[自然场景文本检测识别 - CRAFT - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/141865260)



### 2.1.2 其他

[科普：什么是Faster-RCNN目标检测算法_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1U7411T72r?p=1&vd_source=9d0873d2d65bb6fe93af166e55a35858)

[15. 识别模块网络架构解读_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1gU4y1r7Ku?p=15&vd_source=9d0873d2d65bb6fe93af166e55a35858)

[1. OCR文字识别要完成的任务_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1gU4y1r7Ku?p=1&vd_source=9d0873d2d65bb6fe93af166e55a35858)

## 2.2 文字识别模块

主要分为基于CTC与基于S2S两部分，CTC主要为CRNN与LSTM，S2S主要引入Attention机制

### 2.2.1 CTC

**1.CRNN**

原理：CRNN(Convolutional Recurrent Neural Network）是目前较为流行的图文识别模型，可识别较长的文本序列。它包含CNN特征提取层和BLSTM序列特征提取层，能够进行端到端的联合训练。 它利用BLSTM和CTC部件学习字符图像中的上下文关系， 从而有效提升文本识别准确率，使得模型更加鲁棒。预测过程中，前端使用标准的CNN网络提取文本图像的特征，利用BLSTM将特征向量进行融合以提取字符序列的上下文特征，然后得到每列特征的概率分布，最后通过转录层(CTC rule)进行预测得到文本序列。

[CRNN——卷积循环神经网络结构_循环卷积神经网络-CSDN博客](https://blog.csdn.net/qq_51715775/article/details/115793496?ops_request_misc=%7B%22request%5Fid%22%3A%22170771973316800182131342%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170771973316800182131342&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-115793496-null-null.142^v99^pc_search_result_base5&utm_term=crnn&spm=1018.2226.3001.4187)

[一文读懂CRNN+CTC文字识别 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/43534801)

[文本识别CRNN模型介绍以及pytorch代码实现-CSDN博客](https://blog.csdn.net/weixin_44599230/article/details/125456993?ops_request_misc=%7B%22request%5Fid%22%3A%22170771973316800182131342%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170771973316800182131342&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-125456993-null-null.142^v99^pc_search_result_base5&utm_term=crnn&spm=1018.2226.3001.4187)

两个TensorFlow版本代码已补充在Github Programs文件夹中 CRNN_Tensorflow-master crnn_ctc_ocr_tf-master

代码：

**2.LSTM**

[史上最小白之RNN详解-CSDN博客](https://blog.csdn.net/Tink1995/article/details/104868903?spm=1001.2014.3001.5502)

RNN中经常出现信息无法长距离传播的问题，LSTM是一种典型的RNN设计用于解决长距离传播问题

原理：

[如何从RNN起步，一步一步通俗理解LSTM_rnn lstm-CSDN博客](https://blog.csdn.net/v_JULY_v/article/details/89894058?ops_request_misc=%7B%22request%5Fid%22%3A%22170772045816800186533961%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170772045816800186533961&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-89894058-null-null.142^v99^pc_search_result_base5&utm_term=lstm&spm=1018.2226.3001.4187)

[史上最小白之LSTM 与 GRU详解_gru和lstm模型介绍-CSDN博客](https://blog.csdn.net/Tink1995/article/details/104881633?spm=1001.2014.3001.5502)

[RNN&LSTM中的梯度消失问题 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/359933512)

[详解BiLSTM及代码实现 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/47802053)

[LSTM BiLSTM - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/164545004)

代码已补充在Github Programs文件夹中 Pytorch-NLP-master

### 2.2.2 S2S

[自然场景文本检测识别 - 综述_场景文本识别str-CSDN博客](https://blog.csdn.net/boon_228/article/details/119800300?ops_request_misc=%7B%22request%5Fid%22%3A%22170771732716800192271713%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170771732716800192271713&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-119800300-null-null.142^v99^pc_search_result_base5&utm_term=自然场景文字识别算法&spm=1018.2226.3001.4187)

https://blog.csdn.net/Tink1995/article/details/105012972

### 2.2.3 其他

https://blog.csdn.net/Tink1995/article/details/105012972

## 翻译模块

# 3 模型学习与比较（端到端）

[端到端文本识别算法:CRAFTS（ECCV2020）_crafts文本-CSDN博客](https://blog.csdn.net/qq_39707285/article/details/109072241?ops_request_misc=%7B%22request%5Fid%22%3A%22170771485316800215038027%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170771485316800215038027&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-109072241-null-null.142^v99^pc_search_result_base5&utm_term=crafts文本检测&spm=1018.2226.3001.4187)

# 4 已包含项目简介

**1.Github Programs文件夹中 AttentionOCR-master**

使用**Cascade Mask RCNN**和**Inception Net**的检测方法，Attention LSTM的识别方法

详细文章：

[自然场景文本检测识别 - Attention OCR - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/138589087)

[经典卷积神经网络-Inception（pytorch实现）_inception pytorch-CSDN博客](https://blog.csdn.net/m0_74890428/article/details/127690616?ops_request_misc=%7B%22request%5Fid%22%3A%22170770709516800186557124%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=170770709516800186557124&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-127690616-null-null.142^v99^pc_search_result_base5&utm_term=inception&spm=1018.2226.3001.4187)

**2.Github Programs文件夹中 chineseocr-app**

使用CTPN + YOLO v3 + CRNN结构制作自然场景文字识别

具体网页：[自然场景OCR检测(YOLOv3+CRNN)_yolo ocr-CSDN博客](https://blog.csdn.net/qq_39706357/article/details/88555163?ops_request_misc=&request_id=&biz_id=102&utm_term=yolov文字检测&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-88555163.142^v99^pc_search_result_base5&spm=1018.2226.3001.4187)

**3.Github Programs文件夹中 text-detection-ocr-master**

使用keras和tensorflow基于CTPN+Densenet实现的中文文本检测和识别

# 5 已包含训练数据集

由于数据集较大，只提供下载链接，可用于文本检测和识别领域模型训练的一些大型公开数据集。

### **1.Chinese Text in the Wild(CTW)**

该数据集包含32285张图像，1018402个中文字符(来自于腾讯街景), 包含平面文本，凸起文本，城市文本，农村文本，低亮度文本，远处文本，部分遮挡文本。图像大小2048*2048，数据集大小为31GB。以(8:1:1)的比例将数据集分为训练集(25887张图像，812872个汉字)，测试集(3269张图像，103519个汉字)，验证集(3129张图像，103519个汉字)。

文献链接：https://arxiv.org/pdf/1803.00085.pdf
数据集下载地址：https://ctwdataset.github.io/

### **2.Reading Chinese Text in the Wild(RCTW-17)**

该数据集包含12263张图像，训练集8034张，测试集4229张，共11.4GB。大部分图像由手机相机拍摄，含有少量的屏幕截图，图像中包含中文文本与少量英文文本。图像分辨率大小不等。

下载地址http://mclab.eic.hust.edu.cn/icdar2017chinese/dataset.html
文献：http://arxiv.org/pdf/1708.09585v2

### 3.ICPR MWI 2018 挑战赛

大赛提供20000张图像作为数据集，其中50%作为训练集，50%作为测试集。主要由合成图像，产品描述，网络广告构成。该数据集数据量充分，中英文混合，涵盖数十种字体，字体大小不一，多种版式，背景复杂。文件大小为2GB。

下载地址：
https://tianchi.aliyun.com/competition/information.htm?raceId=231651&_is_login_redirect=true&accounttraceid=595a06c3-7530-4b8a-ad3d-40165e22dbfe

### 4.Total-Text

该数据集共1555张图像，11459文本行，包含水平文本，倾斜文本，弯曲文本。文件大小441MB。大部分为英文文本，少量中文文本。训练集：1255张 测试集：300

下载地址：http://www.cs-chan.com/source/ICDAR2017/totaltext.zip
文献：http:// arxiv.org/pdf/1710.10400v

### 5.Google FSNS(谷歌街景文本数据集)

该数据集是从谷歌法国街景图片上获得的一百多万张街道名字标志，每一张包含同一街道标志牌的不同视角，图像大小为600*150，训练集1044868张，验证集16150张，测试集20404张。

下载地址：http://rrc.cvc.uab.es/?ch=6&com=downloads
文献：http:// arxiv.org/pdf/1702.03970v1

### 6.COCO-TEXT

该数据集，包括63686幅图像，173589个文本实例，包括手写版和打印版，清晰版和非清晰版。文件大小12.58GB，训练集：43686张，测试集：10000张，验证集：10000张

文献: http://arxiv.org/pdf/1601.07140v2
下载地址：https://vision.cornell.edu/se3/coco-text-2/

### 7.Synthetic Data for Text Localisation

在复杂背景下人工合成的自然场景文本数据。包含858750张图像，共7266866个单词实例，28971487个字符，文件大小为41GB。该合成算法，不需要人工标注就可知道文字的label信息和位置信息，可得到大量自然场景文本标注数据。

下载地址：http://www.robots.ox.ac.uk/~vgg/data/scenetext/
文献：http://www.robots.ox.ac.uk/~ankush/textloc.pdf
Code: https://github.com/ankush-me/SynthText (英文版)
Code https://github.com/wang-tf/Chinese_OCR_synthetic_data(中文版)

### 8.Synthetic Word Dataset

下载地址：http://www.robots.ox.ac.uk/~vgg/data/text/

### 9.Caffe-ocr中文合成数据

数据利用中文语料库，通过字体、大小、灰度、模糊、透视、拉伸等变化随机生成，共360万张图片，图像分辨率为280x32，涵盖了汉字、标点、英文、数字共5990个字符。文件大小约为8.6GB

下载地址：https://pan.baidu.com/s/1dFda6R3
