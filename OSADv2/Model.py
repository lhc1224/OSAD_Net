import torch
from OSADv2.gcn import GCN_Model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import math
from bn_lib.nn.modules import SynchronizedBatchNorm2d
from functools import partial
import settings

norm_layer = partial(SynchronizedBatchNorm2d, momentum=settings.BN_MOM)
class APL_module(nn.Module):
    def __init__(self,pose_in_channels, graph_args="mmpose",
                 edge_importance_weighting=True,input_channels=2048,output_channels=256):
        super(APL_module, self).__init__()
        self.gcn_model=GCN_Model(in_channels=pose_in_channels,graph_args=graph_args,
                 edge_importance_weighting=edge_importance_weighting)
        self.support_project=nn.Conv2d(in_channels=input_channels,out_channels=output_channels,
                              kernel_size=3,stride=1,padding=1)
        self.fuse_1=nn.Conv2d(in_channels=output_channels*2,out_channels=output_channels,
                              kernel_size=1,stride=1,padding=0)
        self.obj_project=nn.Conv2d(in_channels=input_channels,out_channels=output_channels,
                             stride=1,kernel_size=1,padding=0)
        self.spatial_project=nn.Conv2d(in_channels=2,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.fuse_2=nn.Conv2d(in_channels=output_channels+20,out_channels=output_channels,stride=1,
                              kernel_size=1,padding=0)
    def forward(self,support_feature,h_box,o_box,pose_input):

        object_map=support_feature * o_box

        object_map = self.obj_project(object_map)
        support_feature=self.support_project(support_feature)
        b,c,h,w=support_feature.size()

        pose=torch.mean(self.gcn_model(pose_input),dim=3)
        pose=pose.view(b,c,1,1)
        pose=pose.repeat(1,1,h,w)
        feature_1=self.fuse_1(torch.cat((support_feature,pose),dim=1))

        area = F.avg_pool2d(o_box, (w,h)) * h * w + 0.0005
        obj=F.adaptive_avg_pool2d(object_map,1)*w*h/area
        #print(area)
        S_obj=(support_feature*obj).view(b,c,h*w)
        S_obj=F.softmax(S_obj,dim=2).view(b,c,h,w)*support_feature
        spatial_feature=self.spatial_project(torch.cat((h_box,o_box),dim=1))
        feature_2=self.fuse_2(torch.cat((S_obj,spatial_feature),dim=1))
        output=feature_1+feature_2
        #output=F.adaptive_avg_pool2d(output,1)

        return output


class MPT_module(nn.Module):
    def __init__(self, input_channels,out_channels,k=128,stage_num=5):
        super(MPT_module, self).__init__()
        self.conv_1=nn.Conv2d(in_channels=input_channels*2,
                              out_channels=out_channels,
                              kernel_size=3,stride=1,padding=1)

        self.stage_num = stage_num

        mu = torch.Tensor(1, input_channels, k).cuda()  ####
        mu.normal_(0, math.sqrt(2. / k))  # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        # self.register_buffer('mu', mu)
        self.mu = mu
        #self.conv1 = nn.Conv2d(c, c, 1)  #######
        #self.conv2 = nn.Sequential(
        #    nn.Conv2d(c, c, 1, bias=False),
        #    norm_layer(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _l2norm(self, inp, dim):
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def forward(self,x,querys,n):

        _,c,h,w=querys.size()
        #b=x.size(0)

        b, c, h, w = x.size()
        x = x.contiguous().view(b, c, h * w)
        mu = self.mu.repeat(b, 1, 1)  # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)  # b * N * c
                z = torch.bmm(x_t, mu)  # b * N * k
                z = F.softmax(z, dim=2)  # b * N * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)  # b * c * k
                mu = self._l2norm(mu, dim=1)

        org_query=querys
        querys=querys.contiguous().view(b,n,c,h,w)
        querys = querys.permute(0, 2, 1, 3, 4)  ## [b,c,n,h,w]
        querys=querys.contiguous().view(b,c,n*h*w)
        with torch.no_grad():
            q_t = querys.permute(0, 2, 1)  # b * N * c
            q_z = torch.bmm(q_t, mu)  # b * N * k
            q_z = F.softmax(q_z, dim=2)  # b * N * k
            #q_z = q_z / (1e-6 + q_z.sum(dim=1, keepdim=True))

        q_z_t = q_z.permute(0, 2, 1)  # b * k * N
        querys = mu.matmul(q_z_t)  # b * c * N
        querys=querys.view(b,c,n,h,w)
        querys= querys.permute(0, 2, 1, 3, 4).contiguous().view(b * n, c, h, w)

        #sum_z=q_z_t.sum(dim=1,keepdim=True).view(b,c,h,w)
        #sum_q = querys.mean(dim=1, keepdim=True)

        querys=torch.cat((org_query,querys),1)

       # querys=torch.cat((query,sum_z),dim=1)

        output=self.conv_1(querys)
        return output,mu

class DCE(nn.Module):
    def __init__(self,input_channels=256,output_channels=256):
        super(DCE, self).__init__()
        self.cosine_eps = 1e-7
        self.conv=nn.Conv2d(in_channels=input_channels*2,out_channels=output_channels,
                            kernel_size=1,stride=1,padding=0)
    def forward(self,querys):
        ### query [b,n,c,h,w]
        b,n,c,h,w=querys.size()

        sim_list=[]
        for i in range(n):
            tmp_q=querys[:,i,:,:,:]   ### [b,c,h,w]
            tmp_qs = torch.zeros([b,n-1,c,h,w]).cuda()
            j=0
            for k in range(n):
                if j==k:
                    continue
                tmp_qs[:,j,:,:,:]=querys[:,k,:,:,:]
                j+=1

            tmp_q=tmp_q.view(b,1,c,h,w).repeat(1,n-1,1,1,1) ### [b,n-1,c,h,w]
            tmp_q=tmp_q.view(-1,c,h*w)     #### [b*(n-1),c,h*w]
            tmp_qs=tmp_qs.view(-1,c,h*w).permute(0,2,1)    ### [b*(n-1),h*w,c]
            tmp_q_norm=torch.norm(tmp_q,2,1,True)     ###
            tmp_qs_norm=torch.norm(tmp_qs,2,2,True)
            similarity=torch.bmm(tmp_qs,tmp_q)/(torch.bmm(tmp_qs_norm,tmp_q_norm)+self.cosine_eps) ### [b*(n-1),h*w,h*w]
            similarity=similarity.max(1)[0].view(-1,h*w)   ### [b*(n-1),h*w]
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                        similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + self.cosine_eps)  ## [b*(n-1),h*w]
            similarity=similarity.view(b,n-1,h,w)
            similarity=torch.mean(similarity,dim=1,keepdim=True) ### [b,1,h,w]
            sim_list.append(similarity*querys[:,i,:,:,:])

        querys_2=torch.stack(sim_list,dim=1)  ### [b,n,c,h,w]


        querys=torch.cat((querys,querys_2),2)

        querys=self.conv(querys.view(b*n,-1,h,w))
        return querys



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



























