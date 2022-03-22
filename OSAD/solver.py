import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam, SGD
from torch.autograd import Variable
from OSAD.os_ad import OS_AD_model
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import torch.nn.functional as F
import math
import time

import os

EPSILON = 1e-8
p = OrderedDict()
EM_MOM = 0.9
p['lr_bone'] = 1e-4  # Learning rate resnet:5e-5, vgg:2e-5
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
lr_decay_epoch = [8, 20, 25]  # [6, 9], now x3 #15

nAveGrad = 10  # Update the weights once in 'nAveGrad' forward passes
showEvery = 50
tmp_path = 'tmp_see'


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.backbone=config.backbone
        self.in_channels=config.in_channels
        self.output_channels=config.output_channels

        self.num_layers = config.n_layers
        self.num_gpu=config.num_GPU
        #self.lr_branch=p['lr_bone']

        self.build_model()

        if self.config.pre_trained:
            self.net_bone.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'train':
            self.net_bone.train()
            print("train")
        else:
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net_bone.load_state_dict(torch.load(self.config.model))
            self.net_bone.eval()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def get_params(self, base_lr):
        ml = []
        for name, module in self.net_bone.named_children():

            if name == 'loss_weight':
                ml.append({'params': module.parameters(), 'lr': p['lr_branch']})
            else:
                ml.append({'params': module.parameters()})
        return ml

    # build the network
    def build_model(self):
        self.net_bone = OS_AD_model(n_layers=50)

        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()
        if self.config.num_GPU>1:
            torch.nn.DataParallel(self.net_bone)

        self.lr_bone = p['lr_bone']


        self.optimizer_bone = Adam([{'params': self.net_bone.extractor[5].parameters(), 'lr': self.lr_bone / 2},
                                    {'params': self.net_bone.extractor[6].parameters(), 'lr': self.lr_bone / 2},
                                    {'params': self.net_bone.extractor[7].parameters(), 'lr': self.lr_bone / 2},
                                    {'params': self.net_bone.cem.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.ptm.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.plm.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.decoder_1.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.decoder_2.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.decoder_3.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.decoder_4.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.dec_5.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.dec_5_2.parameters(), 'lr': self.lr_bone}],
                                   weight_decay=p['wd'])

        self.print_network(self.net_bone, 'trueUnify bone part')

    def test(self, save_path, image_root):

        time_t = 0.0
        self.net_bone.eval()

        for i, data_batch in enumerate(self.test_loader):

            images_, names, im_sizes = data_batch['image'], data_batch['name'], np.asarray(data_batch['size'])
            support_img, obj_mask, per_mask = data_batch['support_img'], data_batch['obj_mask'], data_batch['per_mask']
            with torch.no_grad():

                images = Variable(torch.stack(images_, dim=1)).cuda()
                support_img = Variable(support_img).cuda()
                obj_mask = Variable(obj_mask).cuda()
                per_mask = Variable(per_mask).cuda()
                if len(images.size()) == 4:
                    images = images.unsqueeze(1)
                if self.config.cuda:
                    images = images.cuda()

                time_start = time.time()
                preds, mu = self.net_bone(images, support_img, obj_mask, per_mask)


                torch.cuda.synchronize()
                time_end = time.time()

                time_t = time_t + time_end - time_start
                # for i in range(preds[-1].size(1)):
                pred = np.squeeze(torch.sigmoid(preds[-1][:, 0, :, :, :]).cpu().data.numpy())

                pred = pred / (pred.max() - pred.min() + 1e-12)

                name = names[0][0]

                multi_fuse = 255 * pred

                name_paths = name.split("/")
                str_1 = ""

                save_img_path = os.path.join(save_path, name_paths[0],name_paths[1])


                os.makedirs(save_img_path,exist_ok=True)
                save_img_path=os.path.join(save_img_path, name_paths[-1][:-4] + '.png')


                img = cv2.imread(os.path.join(image_root , name))

                multi_fuse = cv2.resize(multi_fuse, (img.shape[1], img.shape[0]))

                cv2.imwrite(save_img_path, multi_fuse)
                print(save_img_path)

        print("--- %s seconds ---" % (time_t))
        print('Test Done!')

    # training phase
    def train(self, save_path):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        os.makedirs(save_path, exist_ok=True)

        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)

        for epoch in range(self.config.epoch):

            r_edge_loss, r_sal_loss, r_sum_loss = 0, 0, 0
            self.net_bone.zero_grad()

            for i, data_batch in enumerate(self.train_loader):
                images, labels = data_batch['image'], data_batch['label']
                support_img, obj_mask, per_mask = data_batch['support_image'], data_batch['obj_mask'], data_batch['per_mask']

                images = Variable(torch.stack(images, dim=1)).cuda()

                labels = Variable(torch.stack(labels, dim=1)).cuda()

                support_img = Variable(support_img).cuda()
                obj_mask = Variable(obj_mask).cuda()
                per_mask = Variable(per_mask).cuda()

                pred, mu = self.net_bone(images, support_img, obj_mask, per_mask)
                with torch.no_grad():
                    mu = mu.mean(dim=0, keepdim=True)
                    momentum = EM_MOM
                    self.net_bone.cem.mu = momentum
                    self.net_bone.cem.mu += mu * (1 - momentum)

                loss1 = []

                for ix in pred:
                    loss1.append(F.binary_cross_entropy_with_logits(ix, labels, reduction='mean'))

                aff_loss = (sum(loss1)) / (nAveGrad * self.config.batch_size)

                r_sal_loss += aff_loss.data
                loss = aff_loss
                r_sum_loss += loss.data
                loss.backward()
                aveGrad += 1

                if aveGrad % nAveGrad == 0:
                    self.optimizer_bone.step()
                    self.optimizer_bone.zero_grad()
                    aveGrad = 0

                torch.cuda.empty_cache()
                if i % showEvery == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||    Sal : %10.4f  ||  Sum : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num,
                        r_sal_loss * (nAveGrad * self.config.batch_size) / showEvery,
                        r_sum_loss * (nAveGrad * self.config.batch_size) / showEvery))

                    print('Learning rate: ' + str(self.lr_bone))
                    r_edge_loss, r_sal_loss, r_sum_loss = 0, 0, 0

                if i % 100 == 0:
                    vutils.save_image(torch.sigmoid(pred[-1].data[:, 0, :, :, :]), tmp_path + '/iter%d-sal-0.png' % i,
                                      normalize=True, padding=0)

                    vutils.save_image(images.data[:, 0, :, :, :] * 255.0, tmp_path + '/iter%d-sal-data.jpg' % i,
                                      padding=0)

                    vutils.save_image(labels.data[:, 0, :, :, :], tmp_path + '/iter%d-sal-target.png' % i, padding=0)

       
            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(),
                           save_path + 'epoch_%d_bone.pth' % (epoch + 1))

            if epoch in lr_decay_epoch:
                if epoch == 5:
                    self.lr_bone = self.lr_bone * 0.5
                    self.optimizer_bone = Adam([{'params': self.net_bone.extractor[5].parameters(), 'lr': self.lr_bone / 2},
                                    {'params': self.net_bone.extractor[6].parameters(), 'lr': self.lr_bone / 2},
                                    {'params': self.net_bone.extractor[7].parameters(), 'lr': self.lr_bone / 2},
                                    #{'params': self.net_bone.conv_1.parameters(), 'lr': self.lr_bone},
                                    #{'params': self.net_bone.conv_2.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.cem.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.ptm.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.plm.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.decoder_1.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.decoder_2.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.decoder_3.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.decoder_4.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.dec_5.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.dec_5_2.parameters(), 'lr': self.lr_bone}],
                                   weight_decay=p['wd'])

                else:
                    self.lr_bone = self.lr_bone * 0.5
                    self.optimizer_bone = Adam([{'params': self.net_bone.extractor[5].parameters(), 'lr': self.lr_bone / 2},
                                    {'params': self.net_bone.extractor[6].parameters(), 'lr': self.lr_bone / 2},
                                    {'params': self.net_bone.extractor[7].parameters(), 'lr': self.lr_bone / 2},
                                    #{'params': self.net_bone.conv_1.parameters(), 'lr': self.lr_bone},
                                    #{'params': self.net_bone.conv_2.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.cem.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.ptm.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.plm.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.decoder_1.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.decoder_2.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.decoder_3.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.decoder_4.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.dec_5.parameters(), 'lr': self.lr_bone},
                                    {'params': self.net_bone.dec_5_2.parameters(), 'lr': self.lr_bone}],
                                   weight_decay=p['wd'])



