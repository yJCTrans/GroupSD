from __future__ import print_function, absolute_import, division
import time
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import collections
import random
import seaborn as sns
import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F
from .utils.meters import AverageMeter
from matplotlib.colors import BoundaryNorm
from .models import *
from .evaluation_metrics import accuracy
import pdb
from torch.nn.utils import clip_grad_norm_
from .model_complexity import compute_model_complexity
matplotlib.use('Agg')
class Trainer(object):
    def __init__(self, args, model, memory, criterion, num_classes):
        super(Trainer, self).__init__()
        self.model = model
        self.memory = memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.num_classes = num_classes
        self.args = args
        self.sm = nn.Softmax(dim=1)
        self.kl_loss = nn.KLDivLoss(size_average=False)
        self.log_sm = nn.LogSoftmax(dim=1)


    def train(self, epoch, data_loaders, data_loaders_Imagenet, optimizer, print_freq=10, train_iters=400):
        if epoch == 0:  # 只在第一个epoch打印
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            print("\n" + "=" * 50)
            print(f"Model Parameters Report (Epoch {epoch})")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Non-trainable parameters: {total_params - trainable_params:,}")
            print("=" * 50 + "\n")

            print("\n" + "=" * 50)
            num_params, flops = compute_model_complexity(
                self.model, (1, 3, 256, 128)
            )
            print(f"flops: {flops:,}")
            print("=" * 50 + "\n")

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        source_count = len(data_loaders)
        print('source_count={}'.format(source_count))
        end = time.time()
        for i in range(train_iters):

            if True:
                data_loader_index = [i for i in range(source_count)]
                batch_data = [data_loaders[i].next() for i in range(source_count)]

                data_time.update(time.time() - end)

                loss_train = 0.
                for t in data_loader_index: # 0 1 2
                    data_time.update(time.time() - end)

                    traininputs = batch_data[t]
                    inputs, targets = self._parse_data(traininputs)
                    if self.args.SD:
                        f_out, f_out_rb, tri_features, rb_features, out, out_list = self.model(inputs)
                        self._extract_features(out, targets, domain_id=t)

                        alpha_vitkd = 0.00003
                        generation = nn.Sequential(
                            nn.Conv2d(768, 768, kernel_size=3, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(768, 768, kernel_size=3, padding=1))
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        generation.to(device)
                        low_s = out[0]
                        mid_1 = out[1]
                        mid_2 = out[2]
                        high_s = out[3]

                        B = self.args.batch_size
                        loss_mse = nn.MSELoss(reduction='sum')
                        loss_lr_1 = loss_mse(f_out, low_s) / B * alpha_vitkd
                        loss_lr_2 = loss_mse(f_out, mid_1) / B * alpha_vitkd
                        loss_lr_3 = loss_mse(f_out, mid_2) / B * alpha_vitkd
                        loss_lr = loss_lr_1 + loss_lr_2 + loss_lr_3

                        loss_gen = loss_mse(f_out, high_s) / B * alpha_vitkd
                        loss_vitkd = loss_lr + loss_gen

                        loss_s = self.memory[t](f_out, targets).mean()

                        loss_train = loss_train + loss_s + loss_vitkd

                    else:
                        f_out, tri_features = self.model(inputs)
                        loss_s = self.memory[t](f_out, targets).mean()
                        loss_train = loss_train + loss_s


                loss_train = loss_train / source_count

                optimizer.zero_grad()

                loss_train.backward()
                optimizer.step()

                losses.update(loss_train.item())

                with torch.no_grad():
                    for m_ind in range(source_count):
                        imgs, pids = self._parse_data(batch_data[m_ind])
                        if self.args.SD:
                            f_new, _, _, _, _, _ = self.model(imgs)
                        else:
                            f_new, _ = self.model(imgs)
                        self.memory[m_ind].module.MomentumUpdate(f_new, pids)

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Total loss {:.3f} ({:.3f})\t'
                      'loss_s {:.3f}\t'
                      'loss_vitkd {:.3e}\t'

                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              loss_s,
                              loss_vitkd,
                              ))

    def _parse_data(self, inputs):
        imgs, _, pids, _, _ = inputs
        imgs = imgs.cuda()
        pids = pids.cuda()
        return imgs, pids


