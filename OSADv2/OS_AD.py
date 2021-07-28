from OSADv2.Model import APL_module,MPT_module,DCE,Decoder,ConvBNReLU
import torch
from OSADv2.backbone.resnet import resnet
from torchvision.models import  vgg16,densenet121
import torch.nn as nn
import settings
import torch.nn.functional as F

resnet_list=[2048,1024,512,256,128]
class OS_AD(nn.Module):
    def __init__(self,backbone="resnet",in_channels=256,output_channels=256,k=64):
        super(OS_AD, self).__init__()
        if backbone=="resnet":
            self.backbone=resnet(n_layers=50)
            self.extractor = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4)
            self.k=k
            self.conv_1 = nn.Conv2d(in_channels=resnet_list[0], out_channels=in_channels,
                                    kernel_size=3, stride=1, padding=1)
            self.conv_2=nn.Conv2d(in_channels=resnet_list[0],out_channels=in_channels,kernel_size=3,
                                  stride=1,padding=1)


        self.apl = APL_module(pose_in_channels=2,input_channels=in_channels)
        self.mpt = MPT_module(input_channels=in_channels,out_channels=output_channels,k=self.k)
        self.dce = DCE(input_channels=in_channels,output_channels=output_channels)

        self.dec_5 = nn.Sequential(
            ConvBNReLU(in_channels, output_channels, 3, 1, 1, 1),
            nn.Dropout2d(p=0.1))
        self.dec_5_2 = nn.Conv2d(output_channels, 1, 1)
        if backbone=="resnet":
            self.decoder_4 = Decoder(resnet_list[1], output_channels)
            self.decoder_3 = Decoder(resnet_list[2], output_channels)
            self.decoder_2 = Decoder(resnet_list[3], output_channels)
            self.decoder_1 = Decoder(resnet_list[4], output_channels)
        elif backbone=="vgg":
            pass
        else:
            pass

    def forward(self, img,ref,obj_mask,per_mask, pose_data,size=None):
        b,n,img_c,h,w=img.size()
        img=img.view(b*n,img_c,h,w)
        x_1=self.extractor[:4](img)
        x_2=self.extractor[4](x_1)
        x_3=self.extractor[5](x_2)
        x_4=self.extractor[6](x_3)
        x_5=self.conv_1(self.extractor[7](x_4))

        ref=self.conv_2(self.extractor(ref))
        r_b,r_c,r_h,r_w=ref.size()
        obj_h,obj_w=obj_mask.size()[-2:]
        obj_mask=obj_mask.view(b,1,obj_h,obj_w)
        obj_mask=F.interpolate(obj_mask,size=(r_h,r_w),mode='bilinear', align_corners=True)

        per_mask = per_mask.view(b, 1, obj_h, obj_w)
        per_mask=F.interpolate(per_mask,size=(r_h,r_w),mode='bilinear', align_corners=True)

        purpose_encode=self.apl(ref,obj_mask,per_mask,pose_data)

        x_5,mu= self.mpt(purpose_encode, x_5, n)
        x_5_c, x_5_h, x_5_w = x_5.size()[1:]
        x_5=x_5.view(b,n,x_5_c,x_5_h,x_5_w)
        x_5=self.dce(x_5)
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

        return pred,mu
