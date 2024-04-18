import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import open_world_cifar as datasets
import utils
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle
import logging

logger = logging.getLogger(__name__)

def del_tensor_0_cloumn(Cs, device):
    idx = torch.all(Cs[..., :] == 0, axis=1).to(device)
    index=[]
    for i in range(idx.shape[0]):
        if not idx[i].item():
            index.append(i)
    index=torch.tensor(index).to(device)
    Cs = torch.index_select(Cs, 0, index)
    return Cs,index

def un_contrastive_learning(output1, output2, mask, Temperature=0.4):
    out = torch.cat((output1, output2), dim=0)
    sim_mat = torch.mm(out, torch.transpose(out,0,1))
    sim_mat_denom = torch.mm(torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(1).t())
    sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)
    sim_mat = torch.exp(sim_mat / Temperature)
    sim_mat_denom = torch.norm(output1, dim=1) * torch.norm(output2, dim=1)
    sim_match = torch.exp(torch.sum(output1 * output2, dim=-1) / sim_mat_denom / Temperature)
    sim_match = torch.cat((sim_match, sim_match), dim=0)
    mask = torch.cat((mask, mask), dim=0)
    norm_sum = torch.exp(torch.ones(out.size(0)) / Temperature )
    norm_sum = norm_sum.cuda()
    temp = -torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum))
    temp = torch.reshape(temp, (temp.shape[0], 1)).cuda()
    loss = torch.mean(temp * mask)
    return loss

def pl_contrastive_clustering(output1, output2, labels, device, Temperature=0.4):
    out = torch.cat((output1, output2), dim=0)
    labels = labels.contiguous().view(-1)
    labels = torch.cat((labels, labels), dim=0)
    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    sim_mat = torch.mm(out, torch.transpose(out,0,1))
    sim_mat_denom = torch.mm(torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(1).t())
    sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)
    sim_mat = torch.exp(sim_mat / Temperature)
    sim_match = torch.mul(sim_mat,mask)
    norm_sum = torch.exp(torch.ones(out.size(0)) / Temperature)
    norm_sum = norm_sum.cuda()
    pos_num = torch.sum(mask, dim=-1) - 1
    loss = torch.mean(-torch.log((torch.sum(sim_match,dim=-1) - norm_sum) / (torch.sum(sim_mat, dim=-1) - norm_sum)/pos_num))
    return loss

def DKL(_p, _q):
    return  torch.sum(_p * (_p.log() - _q.log()), dim=-1)

def train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, epoch, seen_num, all_num, tf_writer):
    model.train()
    unlabel_loader_iter = cycle(train_unlabel_loader)

    # 1,2 -> weak aug
    # 3 -> strong aug
    for batch_idx, ((x, x2, x3), target) in enumerate(train_label_loader):

        ((ux, ux2, ux3), unlabel_tl) = next(unlabel_loader_iter)

        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        x3 = torch.cat([x3, ux3], 0)
        labeled_len = len(target)
        x, x2, x3, target = x.to(device), x2.to(device), x3.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(x)
        output2 = model(x2)
        output3 = model(x3)

        logit_xw = output2[:labeled_len]
        logit_uxw = output2[labeled_len:]

        prob = F.softmax(output, dim=1)

        logit_weak = torch.cat((logit_xw, logit_uxw), 0)
        pseudo_label = torch.softmax(logit_weak.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        
        targets_u[:labeled_len] = target

        index_seen = targets_u.lt(seen_num).float()
        index_novel = targets_u.ge(seen_num).float()

        # pseudo-labeling thresholds of seen classes and novel classes
        lamda_seen = 0.95
        mask_seen = (index_seen * max_probs).ge(lamda_seen).float()
        lamda_novel = 0.4 + 0.4 * (epoch / 200)
        mask_novel = (index_novel * max_probs).ge(lamda_novel).float()
        mask = mask_seen + mask_novel

        one_mask = torch.ones(labeled_len).to(device)
        mask[:labeled_len] = one_mask

        # Calculate the entropy loss
        entropy_loss = - entropy(torch.mean(prob, 0))

        # Estimate the distribution of labeled and pseudo-labeled data
        p = torch.sum(pseudo_label[mask.bool()],dim = 0)
        p = p/torch.sum(pseudo_label[mask.bool()])
        q = torch.ones(all_num)
        q = (q / all_num).to(device)

        # Calculate the AM loss
        coefficient = DKL(q, p)
        l_AM = MarginLoss(dist = p, s = coefficient)
        label_loss = l_AM(output[:labeled_len], target)
        unlabel_loss = l_AM(output3[labeled_len:], targets_u[labeled_len:], mask[labeled_len:])
        AM_loss = label_loss + unlabel_loss

        # Calculate the PC loss
        mask = torch.reshape(mask, (mask.shape[0], 1)).to(device)
        pc_output1, pc_index1 = del_tensor_0_cloumn(output2 * mask, device)
        pc_output2, pc_index2 = del_tensor_0_cloumn(output3 * mask, device)
        pc_labels = torch.reshape(targets_u, (targets_u.shape[0], 1)).to(device)
        pc_labels = torch.index_select(pc_labels, 0, pc_index1)
        PC_loss = pl_contrastive_clustering(pc_output1, pc_output2, pc_labels, device)

        # Calculate the UC loss
        inv_mask = torch.ones_like(mask) - mask
        inv_mask = torch.reshape(inv_mask, (inv_mask.shape[0], 1)).to(device)
        UC_loss = un_contrastive_learning(output2, output3, inv_mask)

        loss = entropy_loss + AM_loss + PC_loss + UC_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(args, model, labeled_num, device, test_loader, epoch, tf_writer):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    confs = np.array([])
    probs = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
            probs = np.append(probs, prob.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)
    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    overall_acc = cluster_acc(preds, targets)
    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    print(epoch)
    print('Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}'.format(overall_acc, seen_acc, unseen_acc))
    tf_writer.add_scalar('acc/overall', overall_acc, epoch)
    tf_writer.add_scalar('acc/seen', seen_acc, epoch)
    tf_writer.add_scalar('acc/unseen', unseen_acc, epoch)

def main():
    parser = argparse.ArgumentParser(description='LPS')
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--labeled-num', default=50, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='name')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if args.dataset == 'cifar10':
        train_label_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num,
                                                    labeled_ratio=args.labeled_ratio, download=True,
                                                    transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                                      labeled_ratio=args.labeled_ratio, download=True,
                                                      transform=TransformTwice(datasets.dict_transform['cifar_train']),
                                                      unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                             labeled_ratio=args.labeled_ratio, download=True,
                                             transform=datasets.dict_transform['cifar_test'],
                                             unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_label_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=True, labeled_num=args.labeled_num,
                                                     labeled_ratio=args.labeled_ratio, download=True,
                                                     transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                                       labeled_ratio=args.labeled_ratio, download=True,
                                                       transform=TransformTwice(datasets.dict_transform['cifar_train']),
                                                       unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                              labeled_ratio=args.labeled_ratio, download=True,
                                              transform=datasets.dict_transform['cifar_test'],
                                              unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 100
    else:
        warnings.warn('Dataset is not listed')
        return
    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))

    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True,
                                                     num_workers=2, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set,
                                                       batch_size=args.batch_size - labeled_batch_size, shuffle=True,
                                                       num_workers=2, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)

    model = models.resnet18(num_classes=num_classes)
    model = model.to(device)
    if args.dataset == 'cifar10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    elif args.dataset == 'cifar100':
        state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # whether freeze the backbone
    for name, param in model.named_parameters():
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False

    # Set the optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    tf_writer = SummaryWriter(log_dir=args.savedir)

    for epoch in range(args.epochs):
        train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, epoch, args.labeled_num, num_classes, tf_writer)
        test(args, model, args.labeled_num, device, test_loader, epoch, tf_writer)
        scheduler.step()
    tf_writer.close()

if __name__ == '__main__':
    main()
