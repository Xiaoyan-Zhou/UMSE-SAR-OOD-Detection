from __future__ import print_function

import os
import time
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def relu_evidence(y):
    return F.relu(y)

#利用EDL进行不确定性估计
def uncertainty_estimation(model, img_variable, opt):
    output = model(img_variable)
    evidence = relu_evidence(output)
    alpha = evidence + 1
    uncertainty = opt.n_ways / torch.sum(alpha, dim=1, keepdim=True)
    return uncertainty

def validate(val_loader, model, criterion, opt,device):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        with tqdm(val_loader, total=len(val_loader)) as pbar:
            end = time.time()
            for idx, (input, target) in enumerate(pbar):
                if(opt.simclr):
                    input = input[0].float()
                else:
                    input = input.float()
                    
                if torch.cuda.is_available():
                    input = input.to(device)
                    target = target.to(device)

                # compute output
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc2 = accuracy(output, target, topk=(1, 2))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top2.update(acc2[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                
                pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy()), 
                                  "Acc@2":'{0:.2f}'.format(top2.avg.cpu().numpy(),2),
                                  "Loss" :'{0:.2f}'.format(losses.avg,2), 
                                 })

            print('Val_Acc@1 {top1.avg:.3f} Val_Acc@2 {top2.avg:.3f}'
                  .format(top1=top1, top2=top2))

    return top1.avg, top2.avg, losses.avg


def data_test(data_loader, model, device):
    """One epoch validation"""
    batch_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    pro_list = []
    true_label_list = []
    pre_label_list = []
    with torch.no_grad():
        with tqdm(data_loader, total=len(data_loader)) as pbar:
            end = time.time()
            for idx, (input, true_label) in enumerate(pbar):
                if torch.cuda.is_available():
                    input = input.to(device)
                    true_label = true_label.to(device)

                true_label_list.append(true_label.cpu())
                # compute output
                logits = model(input)#logits
                score = F.softmax((logits), dim=1)
                # measure accuracy and record loss
                pro_list.append(score.max().cpu())
                topk = (1,)
                maxk = max(topk)
                _, pred_label = logits.topk(maxk, 1, True, True)
                pre_label_list.append(pred_label[0].cpu())
                acc1, acc2 = accuracy(logits, true_label, topk=(1,2))
                top1.update(acc1[0], input.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                pbar.set_postfix({"Acc@1":'{0:.2f}'.format(top1.avg.cpu().numpy())})
            print('Test_Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, pro_list, true_label_list, pre_label_list

#calculate the uncertainty of data
def data_cal_u(data_loader, model, device, num_classes = 10):
    # switch to evaluate mode
    model.eval()
    true_label_list = []
    to_np = lambda x: x.data.cpu().numpy()
    with torch.no_grad():
        with tqdm(data_loader, total=len(data_loader)) as pbar:
            u_list1 = []
            u_list2 = []
            for idx, (input, true_label) in enumerate(pbar):
                if torch.cuda.is_available():
                    input = input.to(device)
                    true_label = true_label.to(device)

                true_label_list.append(true_label.cpu())
                # compute output
                logits = model(input)  # logits
                evidence = relu_evidence(logits)
                alpha = evidence + 1

                S = torch.sum(alpha, dim=1, keepdim=True)
                u = num_classes / S
                u_list1.append(1-u.cpu()[0][0])
                u_list2.append(1.0/u.cpu()[0][0])
    return u_list1, u_list2


def data_test_logit(data_loader, model, model_f, device, method='MaxLogit', loss_option='CE'):
    """One epoch validation"""

    # switch to evaluate mode
    model.eval()
    true_label_list = []
    to_np = lambda x: x.data.cpu().numpy()
    with torch.no_grad():
        with tqdm(data_loader, total=len(data_loader)) as pbar:
            end = time.time()
            _score1 = []
            _score2 = []
            _probability = []
            _entropy = []
            for idx, (input, true_label) in enumerate(pbar):
                if torch.cuda.is_available():
                    input = input.to(device)
                    true_label = true_label.to(device)

                true_label_list.append(true_label.cpu())
                # compute output
                logits = model(input)#logits

                if loss_option == 'UMSE':
                    evidence = relu_evidence(logits)
                    alpha = evidence + 1
                    probability = alpha / torch.sum(alpha, dim=1, keepdim=True)
                else:
                    probability = F.softmax((logits), dim=1)

                features = model_f(input)
                all_score1 = np.max(to_np(logits), axis=1)#maxlogit
                all_score2 = features.norm(2, dim=1).cpu().numpy()#MaxNorm
                probability_batch = np.max(to_np(probability), axis=1)
                entropy = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
                for i in range(true_label.shape[0]):
                    _score2.append(all_score2[i][0][0])
                    _score1.append(all_score1[i])
                    _probability.append(probability_batch[i])
                    _entropy.append(entropy[i])

    if method == 'MaxLogit':
        return _score1
    elif method == 'DML':
        score = np.array(_score1) + np.array(_score2)
        score = list(score)
        return score
    elif method == 'MaxNorm':
        return _score2
    elif method == 'MSP':
        return _probability
    elif method == 'Energy':
        return _entropy






