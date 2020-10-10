from torch import nn
import os
import torch

# 参数配置,标准的darknet19参数.
cfg = [
    32, 'M', 64, 'M', 128, 64, 128, 'M', 256, 128, 256, 'M', 512, 256, 512,
    256, 512, 'M', 1024, 512, 1024, 512, 1024
]


def make_layers(cfg, in_channels=3, batch_norm=True):
    """
    从配置参数中构建网络
    :param cfg:  参数配置
    :param in_channels: 输入通道数,RGB彩图为3, 灰度图为1
    :param batch_norm:  是否使用批正则化
    :return:
    """
    layers = []
    flag = True  # 用于变换卷积核大小,(True选后面的,False选前面的)
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=v,
                          kernel_size=(1, 3)[flag],
                          stride=1,
                          padding=(0, 1)[flag],
                          bias=False))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            in_channels = v

            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        flag = not flag

    return nn.Sequential(*layers)


class Darknet19(nn.Module):
    """
    Darknet19 模型
    """
    def __init__(self,
                 num_classes=1000,
                 in_channels=3,
                 batch_norm=True,
                 pretrained=False):
        """
        模型结构初始化
        :param num_classes: 最终分类数       (nums of classification.)
        :param in_channels: 输入数据的通道数  (input pic`s channel.)
        :param batch_norm:  是否使用正则化    (use batch_norm, True or False;True by default.)
        :param pretrained:  是否导入预训练参数 (use the pretrained weight)
        """
        super(Darknet19, self).__init__()
        # 调用nake_layers 方法搭建网络
        # (build the network)
        self.features = make_layers(cfg,
                                    in_channels=in_channels,
                                    batch_norm=batch_norm)
        # 网络最后的分类层,使用 [1x1卷积和全局平均池化] 代替全连接层.
        # (use 1x1 Conv and averagepool replace the full connection layer.)
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(output_size=(1)), nn.Softmax(dim=0))
        # 导入预训练模型或初始化
        if pretrained:
            self.load_weight()
        else:
            self._initialize_weights()

    def forward(self, x):
        # 前向传播
        low = self.features[:43](x)
        x = self.features[43:](low)
        return low, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weight(self):
        weight_file = 'weights/darknet19.pth'

        assert len(torch.load(weight_file).keys()) == len(
            self.state_dict().keys())
        dic = {}
        for now_keys, values in zip(self.state_dict().keys(),
                                    torch.load(weight_file).values()):
            dic[now_keys] = values
        self.load_state_dict(dic)


class PassThrougLayer(nn.Module):
    def __init__(self) -> None:
        super(PassThrougLayer, self).__init__()
        self.stride = 2

    def forward(self, x):
        '''
        将26*26*512，转换为13*13*2048
        输入(bach,channel,w,h)
        '''
        B, C, H, W = x.size()
        h = H // self.stride
        w = H // self.stride
        x = x.view(B, C, h, self.stride, w,
                   self.stride).transpose(3, 4).contiguous()
        x = x.view(B, C, h * w,
                   self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(B, C, self.stride * self.stride, h,
                   w).transpose(1, 2).contiguous()
        x = x.view(B, self.stride * self.stride * C, h, w)
        return x


class LeakyConv(nn.Module):
    def __init__(self, ic, oc, k, s=1, p=0) -> None:
        super(LeakyConv,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ic, oc, kernel_size=k, padding=p, stride=s),
            nn.BatchNorm2d(oc), nn.GELU())

    def forward(self, x):
        return self.block(x)


class YOLOv2(nn.Module):
    def __init__(self, classes=20) -> None:
        super(YOLOv2, self).__init__()
        self.classes = classes
        self.darknet = Darknet19(num_classes=1000,pretrained=True)
        self.passthrough = PassThrougLayer()
        '''
        作者在后期的实现中借鉴了ResNet网络，不是直接对高分辨特征图处理，而是增加了一个中间卷积层，先采用64个 $1\times1$ 卷积核进行卷积，然后再进行passthrough处理，这样 $26\times26\times512$的特征图得到 $13\times13\times256$的特征图。这算是实现上的一个小细节。使用Fine-Grained Features之后YOLOv2的性能有1%的提升
        '''

        self.middle_layer = LeakyConv(512, 64, k=1)
        self.extra = nn.Sequential(
            LeakyConv(1024, 1024, k=3, p=1),
            LeakyConv(1024, 1024, k=3, p=1),
        )
        self.transform = nn.Sequential(
            LeakyConv(1024 + 256, 1024, k=3, p=1),
            nn.Conv2d(1024, 5 * (self.classes + 5), kernel_size=1))

    def forward(self, x):
        low, x = self.darknet(x)
        low = self.passthrough(self.middle_layer(low))
        x = self.extra(x)
        x = torch.cat([low, x],dim=1)
        x = self.transform(x)
        return x


if __name__ == '__main__':
    # 原权重文件为1000分类, 在imagenet上进行预训练,
    # Pretrained model train on imagenet dataset. 1000 nums of classifical.
    # top-1 accuracy 76.5% , top-5 accuracy 93.3%.

    net = YOLOv2()
    print(net)
    x = torch.zeros((2, 3, 418, 418))
    out = net(x)
    print(out.size())
