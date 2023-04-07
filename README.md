# UCM_Net
UCM_Net test code
[weights_download](https://drive.google.com/drive/folders/104StVTuM1JY50NKg8M625JMKhx1fcBOp?usp=sharing)
# 详情
## 上传文件
包括4个大区域篡改检测网络和vgg16小区域检测网络（vu2net_doa_2）py文件以及相应权重以及测试代码。
Including 4 large area tamper detection networks and Vgg16 small area detection network (vu2net_doa_2) py files, corresponding weights, and test code.
## 注意事项一
在上述5个网络的代码中可能定义了一些实际没有发挥作用的类或者函数，这是因为在训练前本人忘记删除了，在测试时还请不要修改或者删除，可能会导致无法正确加载权重，如果您重新训练的话可以忽略。
In the code of the above 5 networks, some classes or functions may have been defined that did not actually work. This is because I forgot to delete them before training. Please do not modify or delete them during testing, as it may result in incorrect weight loading. If you retrain, you can ignore them.
## 注意事项二
在大区域篡改与小区域篡改代码中存在使用的自相关计算函数不一致情况，这是由于类Corr比类Self_Correlation_Per效果更好，因为Corr使用ZeroWindow进一步去除了自身与自身的一个相关性，但会占用更多显存，vgg16在初期没有对图片做大程度的下采样，导致需要更多进行自相关计算的算力，所以vu2net_doa_2没有第一层的自相关计算，在之后的自相关计算中使用的也是Self_Correlation_Per，不过在之后多添加了一个卷积层作了一定程度补充。
There is an inconsistency in the use of autocorrelation calculation functions between large-scale and small-scale tampering codes, because the class Corr is better than the class Self_ Correlation_ The Per effect is better because Corr uses ZeroWindow to further remove a correlation between itself and itself, but it will occupy more graphics memory. Vgg16 did not downsample the image to a large extent in the initial stage, which requires more computing power for autocorrelation calculation. Therefore, vu2net_ doa_ 2. There is no first level autocorrelation calculation, and Self is also used in subsequent autocorrelation calculations_ Correlation_ Per, however, an additional convolutional layer was added later to supplement it to some extent.
