# 前列腺分割使用手册 #
#### 软件包 ####
python2.7
numpy
scipy
pyqt
pygraph
sklearn

### 界面 ###
![](http://i.imgur.com/IgL6YQn.png)
#### 显示控件 ####
Segment result：显示分割结果
Ground Truth：显示手动分割结果
New Treatment Image：显示治疗图片
Imge Info:图像大小显示
Train data Info:训练图片文件，文件位置显示
#### 参数控件 ####
Slice：指定显示图像的断层
Iter：模型的迭代次数
TreeNum：随机森林树木的数量
MaxDepth：随机森林树的深度
Sample：采样点的个数
Bootstrap：是否使用Boost方法

#### 操作控件 ####
Segment：执行分割操作
Training：执行训练操作
Load Training Data：执行载入训练图片
Exit：退出平台
Open-Treatment Img:打开治疗图片
Open-Ground Img：打开手动分割图片
Open-Model：打开模型
Save-Model：保存模型
Save-Result：保存分割结果

### 分割步骤 ###
**Step1**
Open-Treatment Img ，载入治疗图片，调节训练参数
![](http://i.imgur.com/xivyvgK.png)
**Step2**
Load Training Data，载入训练图片，点击Training开始训练
![](http://i.imgur.com/rjix8fu.png)
**Step3**
点击Segment，分割治疗图片，分割结束后，显示分割结果
![](http://i.imgur.com/7CHK20d.png)