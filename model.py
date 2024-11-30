#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：model.py
@Author  ：Hao Song
'''
import torch
import torch.nn as nn
from braindecode.models import ShallowFBCSPNet, EEGNetv4

class Branch(nn.Module):
    """
    EEGNetv4 and ShallowFBCSPNet are apis we used from braindecode.models
    detailed parameters are as follows:
    if dataset == 'bcic2a':
        args.n_class = 4
        args.in_channel = 22
        args.input_window_samples = 1120
        args.final_conv_length = 69
    elif dataset == 'bcic2b':
        args.n_class=2
        args.in_channel = 3
        args.input_window_samples = 1020
        args.final_conv_length = 64
    elif dataset == 'physionet':
        args.n_class=4
        args.in_channel = 64
        args.input_window_samples = 645
        args.final_conv_length = 37
    """
    def __init__(self, n_branch, in_channel, n_class, final_conv_length, input_window_samples):
        super(Branch, self).__init__()
        self.label_branch = nn.ModuleList()
        self.n_branch = n_branch
        for i in range(n_branch):
            self.label_branch.append(EEGNetv4(in_chans=in_channel, n_classes=n_class, input_window_samples=input_window_samples))
            self.label_branch.append(ShallowFBCSPNet(in_chans=in_channel, n_classes=n_class, final_conv_length=final_conv_length))

    def forward(self, x):
        outputs = []
        for i in range(self.n_branch):
            outputs.append(self.label_branch[i*2](x))
            outputs.append(self.label_branch[i*2+1](x))
        return outputs

class model(nn.Module):
    def __init__(self, args):
        """
        the score network assign weights to each branch, and each branch is ensembled with EEGNet+ShallowFBCSP,
        so the class of gate is n_branch * 2
        """
        super(model, self).__init__()
        self.args = args
        self.score = EEGNetv4(in_chans=args.in_channel, n_classes=args.n_branch*2, input_window_samples=args.input_window_samples)
        self.branches = Branch(args.n_branch, args.in_channel, args.n_class, args.final_conv_length, args.input_window_samples)

        self.params = [
            {'params': self.score.parameters()},
            {'params': self.branches.parameters()},
        ]

    def forward(self, x):
        score_weight = self.score(x)  # batch_size x n_branch*2
        weight_sm = torch.nn.functional.softmax(score_weight, dim=1)
        weighted = weight_sm.unsqueeze(0).permute(1, 0, 2)  # Gate Mechanism,  batch_size X 1 X n_branch*2

        outputs = self.branches(x)
        outputs = torch.stack(outputs)  # n_branch*2 x batch_size x n_classes
        outputs = outputs.permute(1, 0, 2)  # batch_size x n_branch*2 x n_classes

        weighted_label = torch.bmm(weighted, outputs)  # batch_size x 1 x n_classes
        weighted_label = weighted_label.view(-1, self.args.n_class)  # batch_size x n_classes
        return weighted_label, weight_sm, weighted, outputs # weighted_label is the final output, which should be used to compute loss


    def predict(self, x):
        # if you want to use the gate mechanism
        weighted_label, weight_sm, _, outputs = self.forward(x)
        weighted = weight_sm * (weight_sm > (1 / self.args.n_branch))  # choose weight bigger than avg
        weight_sm = torch.nn.functional.softmax(weighted, dim=1)  # 512x8
        weighted = weight_sm.unsqueeze(0).permute(1, 0, 2)
        weighted_label = torch.bmm(weighted, outputs)
        weighted_label = weighted_label.view(-1, self.args.n_class)
        return weighted_label
