# our method
from __future__ import print_function
import argparse
import os
import torch
from torch.utils.data import DataLoader
from cls_eval import data_test, data_test_logit
import numpy as np
import random
from draw_results import draw_cmatrix, plot_distribution
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
from data_utils import DatasetLoader
from metric_utils import get_measures

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

    parser.add_argument('--data_path', type=str, default=r'../SAR-OOD-Data/MSTAR/SOC/test')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--loss_option', type=str, default='UMSE', choices=['CE', 'UMSE', 'LogitNorm'])
    parser.add_argument('--ood_method', type=str, default='MaxLogit', choices=['MaxLogit', 'DML', 'MaxNorm', 'MSP', 'Energy'])
    parser.add_argument('--model_path', type=str, default=r'./trained_model/resnet18UMSE.pt')
    parser.add_argument('--fig_name', type=str, default='resnet18UMSE')
    parser.add_argument('--num_classes', type=int, default=10)

    opt = parser.parse_args()

    return opt


def OOD_test(opt):
    device = get_device()

    #id data MSTAR
    data_path = r'../SAR-OOD-Data/MSTAR/SOC'
    testset_id = DatasetLoader('test', data_path)
    id_loader = DataLoader(dataset=testset_id, batch_size=opt.batch_size, shuffle=False)  # 测试不确定性的时候将batch_size设置为1

    #ood test SAR SAMPLE
    data_path = r'../SAR-OOD-Data/SAMPLE'
    ood_testset_sample = DatasetLoader('ood', data_path)
    ood_sample_loader = DataLoader(dataset=ood_testset_sample, batch_size=opt.batch_size, shuffle=False)#测试不确定性的时候将batch_size设置为1

    data_path = r'../SAR-OOD-Data/AIRPLANE/SAR-ACD-main'
    ood_testset_airplane=DatasetLoader('ood', data_path)
    ood_airplane_loader = DataLoader(dataset=ood_testset_airplane, batch_size=opt.batch_size, shuffle=False)#测试不确定性的时候将batch_size设置为1

    data_path = r'../SAR-OOD-Data/SHIP/FUSAR-ship'
    ood_testset_ship = DatasetLoader('ood', data_path)
    ood_ship_loader = DataLoader(dataset=ood_testset_ship, batch_size=opt.batch_size,
                                     shuffle=False)  # 测试不确定性的时候将batch_size设置为1

    # model
    # model = torch.load(opt.model_path, map_location=torch.device('cpu'))
    model = torch.load(opt.model_path)
    model_f = torch.nn.Sequential(*list(model.children())[:-1])
    model_f = model_f.to(device)
    model = model.to(device)
    model_f = model_f.to(device)

    test_acc_id, pro_id, true_label_list_id, pre_label_list_id = data_test(id_loader, model, device)
    print('test_acc_id', test_acc_id)
    # Draw matrix
    filename = os.path.join('./results/', opt.fig_name+'.png')
    cmatrix = confusion_matrix(np.array(true_label_list_id), np.array(pre_label_list_id))
    draw_cmatrix(cmatrix, classes=os.listdir(r'../SAR-OOD-Data/MSTAR/SOC/train'), filename=filename, save=True)

    # Score of In Domain
    score_id = data_test_logit(id_loader, model, model_f, device, method=opt.ood_method,
                               loss_option=opt.loss_option)
    # Score of SAMPLES (Different from vehicle categories in MSTAR)
    score_sample = data_test_logit(ood_sample_loader, model, model_f, device, method=opt.ood_method,
                                   loss_option=opt.loss_option)
    # Score of Airplane
    score_airplane = data_test_logit(ood_airplane_loader, model, model_f, device, method=opt.ood_method,
                                     loss_option=opt.loss_option)
    # Score of SHIP
    score_ship = data_test_logit(ood_ship_loader, model, model_f, device, method=opt.ood_method,
                                 loss_option=opt.loss_option)
    tmp1 = np.sum(score_id)
    in_score1_tmp = score_id / tmp1
    out_score_sample_tmp = score_sample / tmp1
    out_score_airplane_tmp = score_airplane / tmp1
    out_score_ship_tmp = score_ship / tmp1
    measures_sample = get_measures(in_score1_tmp, out_score_sample_tmp)
    measures_airplane = get_measures(in_score1_tmp, out_score_airplane_tmp)
    measures_ship = get_measures(in_score1_tmp, out_score_ship_tmp)

    # Draw the figure of Score between different Datasets
    plot_distribution([in_score1_tmp, out_score_sample_tmp, out_score_airplane_tmp, out_score_ship_tmp], ['MSTAR', 'SAMPLE', 'SAR-ACD', 'FUSAR-ship'], savepath=os.path.join('./results/', opt.fig_name + '_'+ opt.ood_method+'.png'))

    print('SAMPLES', round(measures_sample[0], 3), round(measures_sample[1], 3), round(measures_sample[2], 3))
    print('SAR-ACD', round(measures_airplane[0], 3), round(measures_airplane[1], 3), round(measures_airplane[2], 3))
    print('FUSAR-ship', round(measures_ship[0], 3), round(measures_ship[1], 3), round(measures_ship[2], 3))

    return measures_sample, measures_airplane, measures_ship

if __name__ == '__main__':
    opt = parse_option()
    dict_results_sample = {}
    dict_results_SAR_ACD = {}
    dict_results_FUSAR_ship = {}

    print('model_path', opt.model_path)
    print('loss_option', opt.loss_option)

    measures_sample, measures_airplane, measures_ship = OOD_test(opt)
    dict_results_sample[opt.ood_method] = measures_sample
    dict_results_SAR_ACD[opt.ood_method] = measures_airplane
    dict_results_FUSAR_ship[opt.ood_method] = measures_ship
