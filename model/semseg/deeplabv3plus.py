import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()

        if 'resnet' in cfg['backbone']:
            self.backbone = \
                resnet.__dict__[cfg['backbone']](False, multi_grid=cfg['multi_grid'],
                                                 replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations']) # 考虑了多尺度的因素

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

    def forward(self, x, mode="train"):
        h, w = x.shape[-2:] # 图像数据的高和宽，这里默认有标签和无标签图像大小一致

        # imgA_x, imgB_x = torch.split(x, [x.shape[0] // 2, x.shape[0] // 2])
        # featAs = self.backbone.base_forward(imgA_x)
        # featBs = self.backbone.base_forward(imgB_x)
        featx = self.backbone.base_forward(x)

        c1, c4 = featx[0], featx[-1]
        if mode == "eval":
            c1A_x, c1B_x = c1.chunk(2)
            c4A_x, c4B_x = c4.chunk(2)
            out_x = self._decode(self._difference(c1A_x, c1B_x), self._difference(c4A_x, c4B_x))
            out_x = F.interpolate(out_x, size=(h, w), mode="bilinear", align_corners=True)
            return out_x
            
        c1A_x, c1B_x, c1A_u_w, c1B_u_w, c1A_u_s1, c1B_u_s1, c1A_u_s2, c1B_u_s2 = c1.chunk(8)
        c4A_x, c4B_x, c4A_u_w, c4B_u_w, c4A_u_s1, c4B_u_s1, c4A_u_s2, c4B_u_s2 = c4.chunk(8)

        out_x = self._decode(self._difference(c1A_x, c1B_x), self._difference(c4A_x, c4B_x))
        out_x = F.interpolate(out_x, size=(h, w), mode="bilinear", align_corners=True)
        
        c1_w = self._difference(c1A_u_w, c1B_u_w)
        c4_w = self._difference(c4A_u_w, c4B_u_w)
        out_w = self._decode(
            torch.cat((c1_w, nn.Dropout2d((0.5))(c1_w))),
            torch.cat((c4_w, nn.Dropout2d((0.5))(c4_w))),
        )
        out_w = F.interpolate(out_w, size=(h, w), mode="bilinear", align_corners=True)
        out_u_w, out_u_fp = out_w.chunk(2)
        
        c1_s1 = self._difference(c1A_u_s1, c1B_u_s1)
        c4_s1 = self._difference(c4A_u_s1, c4B_u_s1)
        out_s1 = self._decode(c1_s1, c4_s1)
        out_s1 = F.interpolate(out_s1, size=(h, w), mode="bilinear", align_corners=True)
        
        c1_s2 = self._difference(c1A_u_s2, c1B_u_s2)
        c4_s2 = self._difference(c4A_u_s2, c4B_u_s2)
        out_s2 = self._decode(c1_s2, c4_s2)
        out_s2 = F.interpolate(out_s2, size=(h, w), mode="bilinear", align_corners=True)
        
        return out_x, out_u_w, out_u_fp, out_s1, out_s2
        # c1 = self._difference(cA1, cB1)
        # c4 = self._difference(cA4, cB4)

        # if need_fp:
        #     outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
        #                         torch.cat((c4, nn.Dropout2d(0.5)(c4))))
        #     outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
        #     out, out_fp = outs.chunk(2)

        #     return out, out_fp

        # out = self._decode(c1, c4)
        # out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        # return out

    def _decode(self, c1, c4):
        # c1 是浅层特征      c4 是深层特征
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True) # 上采样特征图的大小

        c1 = self.reduce(c1) # 缩减通道数目

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out

    def _difference(self, imgA, imgB):
        return torch.abs(imgA - imgB)



def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1) # 使用空洞卷积，扩大了感受野
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
