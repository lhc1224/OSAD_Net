from functools import partial
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from bn_lib.nn.modules import SynchronizedBatchNorm2d
from OSAD import settings

norm_layer = partial(SynchronizedBatchNorm2d, momentum=settings.BN_MOM)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, dilation, dilation,
                               bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, stride=8):
        self.inplanes = 128
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False))

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        if stride == 16:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=1, dilation=2, grids=[1, 2, 4])
        elif stride == 8:
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=1, dilation=2)
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=1, dilation=4, grids=[1, 2, 4])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    grids=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))

        layers = []
        if grids is None:
            grids = [1] * blocks

        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                downsample=downsample,
                                previous_dilation=dilation))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample,
                                previous_dilation=dilation))
        else:
            raise RuntimeError('=> unknown dilation size: {}'.format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=dilation * grids[i],
                                previous_dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(n_layers, stride):
    layers = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }[n_layers]
    pretrained_path = {
        50: '/home/lhc/EMAnet_models/models/resnet50-ebb6acbb.pth',
        101: './models/resnet101-2a57e44d.pth',
        152: './models/resnet152-0d43d698.pth',
    }[n_layers]

    net = ResNet(Bottleneck, layers=layers, stride=stride)
    state_dict = torch.load(pretrained_path)
    net.load_state_dict(state_dict, strict=False)

    return net


class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CEM(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''

    def __init__(self, c, k, stage_num=5):
        super(CEM, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k).cuda()    ####
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        #self.register_buffer('mu', mu)
        self.mu=mu
        self.conv1 = nn.Conv2d(c, c, 1)   #######
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        #### x[B,N,C,H,W]
        # The first 1x1 conv

        b,n, c, h, w = x.size()
        x=x.view(b*n,c,h,w)
        idn = x
        x = self.conv1(x)   #### [b*n,c,h,w]


        # The EM Attention
        #b, c, h, w = x.size()
        x=x.contiguous().view(b,n,c,h,w)

        x=x.permute(0,2,1,3,4)   ### x[b,c,n,h,w]
        x=x.contiguous().view(b,c,n*h*w)  ### [b,c,N]
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * N * c
                z = torch.bmm(x_t, mu)  # b * N * k
                z = F.softmax(z, dim=2)  # b * N * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        # !!! The moving averaging operation is writtern in train.py, which is significant.

        z_t = z.permute(0, 2, 1)  # b * k * N
        x = mu.matmul(z_t)  # b * c * N
        x = x.view(b, c,n, h, w)  # b * c *n* h * w

        x = F.relu(x, inplace=True)
        x = x.permute(0, 2, 1, 3, 4).view(b*n,c,h,w)

        # The second 1x1 conv
        x = self.conv2(x)

        x = F.relu(x + idn,inplace=True)
        x = x.view(b, n, c, h, w)


        return x, mu

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))


class  OS_AD_model(nn.Module):

    def __init__(self, n_layers=50):
        super().__init__()
        backbone = resnet(n_layers, settings.STRIDE)
        self.extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4)
        self.plm=PLM(input_channels=2048)
        self.ptm = PTM(channels=2048)
        self.cem = CEM(512, 256, settings.STAGE_NUM)

        self.dec_5 = nn.Sequential(
            ConvBNReLU(512, 256, 3, 1, 1, 1),
            nn.Dropout2d(p=0.1))
        self.dec_5_2 = nn.Conv2d(256, 1, 1)

        self.decoder_4=Decoder(1024,256)
        self.decoder_3=Decoder(512,256)
        self.decoder_2=Decoder(256,256)
        self.decoder_1=Decoder(128,256)



    def forward(self, img,ref,obj_mask,per_mask, lbl=None, size=None):
        b,n,img_c,h,w=img.size()
        img=img.view(b*n,img_c,h,w)
        x_1=self.extractor[:4](img)
        x_2=self.extractor[4](x_1)
        x_3=self.extractor[5](x_2)
        x_4=self.extractor[6](x_3)
        x_5=self.extractor[7](x_4)

        ref=self.extractor(ref)
        r_b,r_c,r_h,r_w=ref.size()
        obj_h,obj_w=obj_mask.size()[-2:]
        obj_mask=obj_mask.view(1,1,obj_h,obj_w)
        obj_mask=F.interpolate(obj_mask,size=(r_h,r_w),mode='bilinear', align_corners=True)
        obj_mask=obj_mask*ref

        per_mask = per_mask.view(1, 1, obj_h, obj_w)
        per_mask=F.interpolate(per_mask,size=(r_h,r_w),mode='bilinear', align_corners=True)
        per_mask=per_mask*ref
        purpose_encode=self.plm(ref,obj_mask,per_mask)

        x_5=self.ptm(x_5,purpose_encode,b,n)

        x_5_c, x_5_h, x_5_w = x_5.size()[1:]
        x_5=x_5.view(b,n,x_5_c,x_5_h,x_5_w)
        x_5,mu=self.cem(x_5)
        x_5 = x_5.view(b * n, x_5_c, x_5_h, x_5_w)

        tmp_x_5=self.dec_5(x_5)
        x_5 = self.dec_5_2(tmp_x_5)

        tmp_x_4,x_4=self.decoder_4(tmp_x_5,x_4)
        tmp_x_3,x_3=self.decoder_3(tmp_x_4,x_3)
        tmp_x_2,x_2=self.decoder_2(tmp_x_3,x_2)
        tmp_x_1,x_1=self.decoder_1(tmp_x_2,x_1)

        pred = []
        if size is None:
            size = img.size()[-2:]

        pred.append(F.interpolate(x_5, size=size, mode='bilinear', align_corners=True).view(b, n, -1, size[0], size[1]))

        pred.append(F.interpolate(x_4, size=size, mode='bilinear', align_corners=True).view(b, n, -1, size[0], size[1]))

        pred.append(F.interpolate(x_3, size=size, mode='bilinear', align_corners=True).view(b, n, -1, size[0], size[1]))

        pred.append(F.interpolate(x_2, size=size, mode='bilinear', align_corners=True).view(b, n, -1, size[0], size[1]))
        pred.append(F.interpolate(x_1, size=size, mode='bilinear', align_corners=True).view(b, n, -1, size[0], size[1]))

        return pred, mu




class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='none', ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, reduction=reduction,
                                   ignore_index=ignore_index)

    def forward(self, inputs, targets):
        loss = self.nll_loss(F.log_softmax(inputs, dim=1), targets)
        return loss.mean(dim=2).mean(dim=1)


def test_net():
    model = OS_AD_model(n_layers=50)
    model.eval()
    print(list(model.named_children()))
    image = torch.randn(1, 3, 513, 513)
    label = torch.zeros(1, 513, 513).long()
    pred = model(image, label)
    print(pred.size())

class PLM(nn.Module):
    def __init__(self,input_channels=2048):
        super().__init__()

        self.input_channels=input_channels
        self.pro=nn.Conv2d(self.input_channels,512,kernel_size=1,stride=1,padding=0)
        self.obj_pro=nn.Conv2d(self.input_channels,512,kernel_size=3,stride=1,padding=1)
        self.human_pro=nn.Conv2d(self.input_channels,512,kernel_size=3,stride=1,padding=1)
        self.conv=nn.Conv2d(512,1,kernel_size=3,stride=1,padding=1)


    def forward(self, feature_map,obj,human):
        feature_map = self.pro(feature_map)
        obj_pro = self.obj_pro(obj)
        human_pro = self.human_pro(human)

        obj = F.adaptive_max_pool2d(obj_pro, 1)
        human = F.adaptive_max_pool2d(human_pro, 1)

        b, c, h, w = feature_map.size()
        F_obj = F.softmax((obj * feature_map).view(b * c, h * w), dim=1).view(b, c, h, w)
        F_obj = F_obj * feature_map

        F_human = F.softmax((human * feature_map).view(b * c, h * w), dim=1).view(b, c, h, w)
        F_human = F_human * feature_map



        obj_h=self.conv(obj*human_pro)
        S_o_att=F_obj*obj_h

        S_h_att = F_human*obj_h


        output=F.adaptive_max_pool2d(S_h_att+S_o_att,1)

        return output

class PTM(nn.Module):
    def __init__(self,channels=2048):
        super().__init__()

        #self.input_channels=channels


        self.fc1=ConvBNReLU(channels, 512, 3, 1, 1, 1)
        self.fc2=nn.Conv2d(512,512,kernel_size=3,padding=1,stride=1)


    def forward(self, x,purpose_encode,b,n):
        x = self.fc1(x)

        x_c, x_h, x_w = x.size()[1:]
        purpose_encode=purpose_encode.contiguous().view(b,x_c,1,1)
        purpose_encode=purpose_encode.repeat(n,1,x_h,x_w)
        purpose_encode = x * purpose_encode
        purpose_encode = F.softmax(purpose_encode.view(b * n * x_c, x_h * x_w), dim=1)
        x = x + purpose_encode.view(b * n, x_c, x_h, x_w) * x
        output = self.fc2(x)
        return output
class Decoder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()

        self.fc1=ConvBNReLU(input_channels, output_channels, 3, 1, 1, 1)
        self.fc2=nn.Sequential(ConvBNReLU(output_channels, output_channels, 3, 1, 1, 1),
                               nn.Dropout2d(p=0.1))
        self.fc3=nn.Conv2d(output_channels, 1, 1)

    def forward(self, x0,x):
        x=self.fc1(x)
        x = self.fc2(x + F.interpolate(x0, size=x.size()[-2:], mode='bilinear', align_corners=True))
        output=self.fc3(x)
        return  x,output
