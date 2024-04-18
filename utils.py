import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import os.path
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from randaugment import RandAugmentMC
from randaugment import RandAugmentMC_imagenet



class TransformTwice:
    #  cifar100
    def __init__(self, transform, test = 0):
        self.owssl = transform
        self.test = test

        self.simclr = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.fixmatch_weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.fixmatch_strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    def __call__(self, inp):
        out1 = self.owssl(inp)
        out2 = self.owssl(inp)
        # out3 = self.simclr(inp)
        # out4 = self.simclr(inp)
        #out5 = self.fixmatch_weak(inp)
        out6 = self.fixmatch_strong(inp)
        if self.test == 0:
            return out1, out2 , out6
        if self.test == 1:
            return out1, out2


class TransformTwice_imagenet:
    #  cifar100
    def __init__(self, transform, test = 0):
        self.owssl = transform
        self.test = test

        self.fixmatch_weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224*0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.fixmatch_strong = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(224 * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC_imagenet(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, inp):
        out1 = self.owssl(inp)
        out2 = self.owssl(inp)
        out3 = self.fixmatch_strong(inp)
        if self.test == 0:
            return out1, out2 , out3
        if self.test == 1:
            return out1, out2


class AverageMeter(object):
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target):
    
    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res

def cluster_acc(y_pred, y_true):

    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size


def entropy(x):

    EPS = 1e-8
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class MarginLoss(nn.Module):
    
    def __init__(self, dist=None, weight=None, s=1):
        super(MarginLoss, self).__init__()
        self.dist = (dist * (10 / dist.max())).to(torch.device("cuda"))
        self.s = s
        self.weight = weight

    def forward(self, x, target, mask=None):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index2 = torch.zeros_like(x, dtype=torch.float)
        index2.scatter_(1, target.data.view(-1, 1), 1)
        
        batch_m = torch.matmul(self.dist[None, :], torch.transpose(index2,0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x + self.s * batch_m
        output = torch.where(index, x_m, x)
        if mask == None:
            return F.cross_entropy(output, target, weight=self.weight)
        return (F.cross_entropy(output, target, weight=self.weight, reduction='none') * mask).mean()


