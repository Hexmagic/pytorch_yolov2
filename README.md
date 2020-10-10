最小化实现YoloV2
## 安装
克隆代码，安装包：
```
git clone ...
pip install -r requirements.txt
```
## 数据集
下载数据集并解压
```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
## 训练

需要先下载darknet预训练权重，下载地址：
```
https://github.com/yatengLG/darknet19-pytorch/blob/master/weights/darknet19-deepBakSu-e1b3ec1e.pth
```
百度云链接: https://pan.baidu.com/s/18_OMLuy6cN56HmZaQYQzvg  密码: rw52

下载后存在weights文件夹下(没有该文件夹就新建)，直接执行训练脚本即可
```
python train.py --batch_size 16
```
训练获得的权重存在weights下面

## Demo

直接运行demo.py指定预训练权重即可查看结果
```
python demo.py --model weights/yolov2_160.pth
```