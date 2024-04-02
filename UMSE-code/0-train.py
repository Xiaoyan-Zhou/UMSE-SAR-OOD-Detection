# our method
from __future__ import print_function

import argparse
import os.path
import time
import torchvision.models
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from util import accuracy, AverageMeter
from cls_eval import data_test
import numpy as np
import copy
import random
from loss import edl_mse_loss, LogitNormLoss
import torch.nn.functional as F
from data_utils import DatasetLoader

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--loss_option', type=str, default='UMSE', choices=['CE', 'UMSE', 'LogitNorm'],
                        help='the choice of loss function')

    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'wrn', 'densenet'], help='the choice of model')
    # optimization for Adam
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay')

    # dataset and model
    parser.add_argument('--data_path', type=str, default=r'../SAR-OOD-Data/MSTAR/SOC')
    parser.add_argument('--model_path', type=str, default=r'./trained_model/', help='path for saving trained models')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes for classification')#parameter for support set

    opt = parser.parse_args()

    return opt

def one_hot_embedding(labels, num_classes=3):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

def y_embedding(y_hat,y):
    # Convert to One Hot Encoding
    y.append(y_hat)
    return y

def relu_evidence(y):
    return F.relu(y)

def train(train_loader, model, criterion_cls, optimizer, opt, epoch, device):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for i, (input, label) in enumerate(pbar):
            # train_count += 1
            data_shot = input.to(device)
            logits = model(data_shot)
            if opt.loss_option == 'UMSE':#use uncertainty aware loss function
                y = one_hot_embedding(label, opt.num_classes)#one_hot 向量
                y = y.to(device)
                label = label.to(device)
                loss1 = criterion_cls(logits, y.float(), epoch, opt.num_classes, 5, device, KL=False)
            elif opt.loss_option == 'LogitNorm':
                label = label.to(device)
                loss1 = criterion_cls(logits, label)
            elif opt.loss_option == 'CE':
                label = label.to(device)
                loss1 = criterion_cls(logits, label)
            loss = loss1
            acc1, acc2 = accuracy(logits, label, topk=(1, 2))
            top1.update(acc1[0], input.size(0))
            losses.update(loss.item(), input.size(0))
            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            pbar.set_postfix({"Acc@1": '{0:.2f}'.format(top1.avg.cpu().numpy()),
                              "Loss": '{0:.2f}'.format(losses.avg, 2),
                              })

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg


def main(opt):
    device = get_device()

    # loss function
    if opt.loss_option =='UMSE':
        criterion_cls = edl_mse_loss
        print('the loss function is UMSE')
    elif opt.loss_option == 'LogitNorm':
        criterion_cls = LogitNormLoss(device, t=0.01)
    elif opt.loss_option == 'CE':
        criterion_cls = nn.CrossEntropyLoss()
        print('the loss function is cross Entropy')

    trainset = DatasetLoader('train', opt.data_path)#用随机种子保证每次采样得到的数据集不同
    train_loader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
    # iid test the angle of test set is the same as train set
    testset = DatasetLoader('test', opt.data_path)
    test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False)

    # model
    if opt.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False, num_classes=opt.num_classes)
    elif opt.model == 'wrn':
        # model = WideResNet(40, opt.num_classes, widen_factor=2, dropRate=0.3, use_norm=False, feature_norm=False) #WRN-40-2
        model = torchvision.models.wide_resnet50_2(pretrained=False, num_classes=opt.num_classes)
    elif opt.model == 'densenet':
        model = torchvision.models.densenet121(pretrained=False, num_classes=opt.num_classes)

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    model = model.to(device)
    test_acc = 0
    # routine: supervised model distillation
    best_model_test = copy.deepcopy(model)
    for epoch in range(1, opt.epochs + 1):
        print("==> training...")
        time1 = time.time()
        train_acc, train_loss = train(train_loader, model, criterion_cls,
                                      optimizer, opt, epoch, device)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        print('trainning loss', train_loss, 'train_acc', train_acc)
        if epoch % 3 == 0:
            test_acc_new, _, _, _ = data_test(test_loader, model, device)
            print('===> Test Acc', test_acc_new)
            if test_acc < test_acc_new and epoch > 45:
                test_acc = test_acc_new
                best_model_test = copy.deepcopy(model)
    test_acc, _, _, _ = data_test(test_loader, best_model_test, device)
    torch.save(best_model_test, os.path.join(opt.model_path, '{}_1.pt'.format(opt.model + opt.loss_option)))

    print('the best acc is:', test_acc)

if __name__ == '__main__':
    opt = parse_option()
    setup_seed(3407)
    main(opt)

