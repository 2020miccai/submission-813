import time
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.optim as optm
import torch.nn.functional as func
import matplotlib.pyplot as plt
import pdb
import scipy.io as sio
from torch.autograd import Variable

from os.path import join
from tqdm import tqdm
from shutil import copyfile
from domainadapt.input.graphLoader_sub import GeometricDataset
from domainadapt.input.dataloader import DataLoader
from domainadapt.custom_callbacks.Loss_plotter import LossPlotter
from domainadapt.custom_callbacks.Logger import Logger
from domainadapt.models.model_seg import GCNSeg
from domainadapt.models.model_dis import GCNDis
from domainadapt.nn.criterions import CombDiceCross
from domainadapt.utils.utils import compute_dice_metrics, compute_acc_metrics
sys.path.insert(0, './domainadapt/input/')

def _get_config():
    parser = argparse.ArgumentParser(description="Main handler for training",
                                     usage="python ./train.py -j config.json -g 0")
    parser.add_argument("-j", "--json", help="configuration json file", required=True)
    parser.add_argument('-g', '--gpu', help='Cuda Visible Devices', required=True)
    args = parser.parse_args()

    with open(args.json, 'r') as f:
        config = json.loads(f.read())

    initial_weights = config['generator']['initial_epoch']
    directory = os.path.join(config['directories']['out_dir'],
                             config['directories']['ConfigName'],
                             'config', str(initial_weights))
    if not os.path.exists(directory):
        os.makedirs(directory)

    copyfile(args.json, os.path.join(config['directories']['out_dir'],
                                     config['directories']['ConfigName'],
                                     'config', str(initial_weights), 'config.json'))

    # Set the GPU flag to run the code
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return config


def main(config):
    device = torch.device("cuda")

    generator_config = config['generator']  # Model experiments total epochs and beginning epoch
    initial_epoch = generator_config['initial_epoch']  # O by default and otherwise 'N' if loading
    num_epochs = generator_config['num_epochs']  # Total Number of Epochs
    plt_sep = generator_config['plot_separate']  # Plot the train, valid and test separately: 0 or 1
    set_lamda = generator_config['set_lamda']  # Plot the train, valid and test separately: 0 or 1
    plot_hist = generator_config['plot_hist']  # Plot the histogram of the parcel size

    model_params = config['model_params']
    feat = model_params['feat']
    hid = model_params['hid']
    par = model_params['par']
    emb_siz = model_params['emb_siz']
    ker_siz = model_params['ker_siz']

    feat_dis = model_params['feat_dis']
    hid_dis = model_params['hid_dis']
    par_dis = model_params['par_dis']
    emb_siz_dis = model_params['emb_siz_dis']
    ker_siz_dis = model_params['ker_siz_dis']

    # Train with Weights: Used to give priority for smaller sized parcels
    if model_params['use_weight']:
        wht = torch.empty(1, par, dtype=torch.float)
        wht.data = torch.tensor([16.97, 85.77, 32.16, 43.73, 232.04, 31.40, 20.52, 26.56,
                                 99.08, 17.44, 32.52, 29.09, 65.74, 21.93, 136.30, 54.85,
                                 60.93, 126.48, 60.71, 67.27, 18.68, 69.16, 18.31, 24.79,
                                 80.78, 21.33, 11.76, 20.24, 18.45, 24.17, 184.53, 47.92]).to(device)
    else:
        wht = torch.ones(par, dtype=torch.float).to(device)

    if set_lamda:
        lda = "global"
        lda = torch.empty(1, dtype=torch.float)
        lda.data = torch.tensor([set_lamda]).to(device)
        alpha = 'global'
        alpha = torch.empty(1, dtype=torch.float)
        alpha.data = torch.tensor([1]).to(device)
    else:
        lda = "global"
        lda = torch.empty(1, dtype=torch.float)
        lda.data = torch.tensor([1]).to(device)

    optm_config = config['optimizer_wh']
    b1 = optm_config['B1']  # B1 for Adam Optimizer: Ex. 0.9
    b2 = optm_config['B2']  # B2 for Adam Optimizer: Ex. 0.999
    lr_wh_seg = optm_config['LR_seg']  # Learning Rate: Ex. 0.001
    lr_wh_dis = optm_config['LR_dis']  # Learning Rate: Ex. 0.001
    optm_con_mu = config['optimizer_mu']
    lr_mu = optm_con_mu['LR']  # Learning Rate: Ex. 0.001
    optm_con_si = config['optimizer_si']
    lr_si = optm_con_si['LR']  # Learning Rate: Ex. 0.001
    directory_config = config['directories']
    out_dir = directory_config['out_dir']  # Path to save the outputs of the experiments
    config_name = directory_config['ConfigName']  # Configuration Name to Uniquely Identify this Experiment
    log_path = join(out_dir, config_name, 'log')  # Path to save the training log files
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(join(log_path, 'weights')):  # Path to save the weights of training
        os.makedirs(join(log_path, 'weights'))
    main_path = directory_config['datafile']  # Full Path of the dataset. Folder contains train, valid and test

    # Initialize the model, optimizer and data loader
    model_seg = GCNSeg(feat=feat, hid=hid, par=par, emb_size=emb_siz, ker_size=ker_siz)  # Create the model
    model_seg = model_seg.to(device)

    model_dis = GCNDis(feat=feat_dis, hid=hid_dis, par=par_dis, emb_size=emb_siz_dis, ker_size=ker_siz_dis)
    model_dis = model_dis.to(device)

    compute_loss = CombDiceCross()  # Loss function: Here combining weighted dice and weighted cross-entropy
    bce_loss = torch.nn.BCELoss()

    # optimizer_mu : updates 'mu' parameters of the network
    # optimizer_si: updates 'sigma' parameters of the network
    # optimizer_wh: updates 'weights' and bias parameters of the network

    optimizer_seg_mu = optm.SGD([model_seg.gc1_1.mu, model_seg.gc2_1.mu, model_seg.gc3_1.mu], lr=lr_mu)
    optimizer_seg_si = optm.SGD([model_seg.gc1_1.sig, model_seg.gc2_1.sig, model_seg.gc3_1.sig], lr=lr_si)
    optimizer_seg_wh = optm.Adam([model_seg.gc1_1.weight, model_seg.gc2_1.weight, model_seg.gc3_1.weight,
                                  model_seg.gc1_1.bias, model_seg.gc2_1.bias, model_seg.gc3_1.bias], lr=lr_wh_seg,
                                 betas=(b1, b2))

    optimizer_dis_mu = optm.SGD([model_dis.gc1_1.mu, model_dis.gc2_1.mu], lr=lr_mu)
    optimizer_dis_si = optm.SGD([model_dis.gc1_1.sig, model_dis.gc2_1.sig], lr=lr_si)
    optimizer_dis_wh = optm.Adam(
        [model_dis.gc1_1.weight, model_dis.gc2_1.weight, model_dis.lin1.weight, model_dis.lin2.weight,
         model_dis.lin3.weight,
         model_dis.gc1_1.bias, model_dis.gc2_1.bias, model_dis.lin1.bias, model_dis.lin2.bias, model_dis.lin3.bias],
        lr=lr_wh_dis,
        betas=(b1, b2))

    # train_lb_set = GeometricDataset('train_dm1', main_path)
    # train_lb_loader = DataLoader(train_lb_set,
    #                              batch_size=generator_config['batch_size'],
    #                              num_workers=4,
    #                              shuffle=True)
    #
    # train_un_set = GeometricDataset('train_dm2', main_path)
    # train_un_loader = DataLoader(train_un_set,
    #                              batch_size=generator_config['batch_size'],
    #                              num_workers=4,
    #                              shuffle=True)
    #
    # valid_set = GeometricDataset('val', main_path)
    # valid_loader = DataLoader(valid_set,
    #                           batch_size=generator_config['batch_size'],
    #                           num_workers=4,
    #                           shuffle=False)
    #
    # test_lb_set = GeometricDataset('test_dm1', main_path)
    # test_lb_loader = DataLoader(test_lb_set,
    #                             batch_size=generator_config['batch_size'],
    #                             num_workers=4,
    #                             shuffle=False)

    test_un_set = GeometricDataset('test_dm2', main_path)
    test_un_loader = DataLoader(test_un_set,
                                batch_size=generator_config['batch_size'],
                                num_workers=4,
                                shuffle=False)

    if initial_epoch > 0:
        print("===> Loading pre-trained weight {}".format(initial_epoch))
        weight_path = 'weights/model-{:04d}.pt'.format(initial_epoch)
        checkpoint = torch.load(join(log_path, weight_path))
        model_seg.load_state_dict(checkpoint['model_state_dict'])
        # optm.load_state_dict(checkpoint['optimizer_state_dict'])

    def checkpoint(epc):
        w_path = 'weights/model-{:04d}.pt'.format(epc)
        torch.save(
            {'epoch': epc, 'model_state_dict': model_seg.state_dict(),
             'optimizer_state_dict': [optimizer_seg_wh.state_dict(),
                                      optimizer_seg_mu.state_dict(),
                                      optimizer_seg_si.state_dict(),
                                      optimizer_dis_wh.state_dict(),
                                      optimizer_dis_mu.state_dict(),
                                      optimizer_dis_si.state_dict()]}, join(log_path, w_path))

    # setup our callbacks
    my_metric = ['Dice', 'Segmentation_Accuracy', 'Discriminator_Acc']
    my_loss = ['Loss_segmentator', 'Loss_seg_adv', 'Loss_seg_lab', 'Loss_discriminator', 'Loss_dis_lab', 'Loss_dis_unl']
    source_label = Variable(torch.FloatTensor([0])).to(device)
    target_label = Variable(torch.FloatTensor([1])).to(device)
    logger = Logger(mylog_path=log_path, mylog_name="training.log", myloss_names=my_loss, mymetric_names=my_metric)
    ls_plt = LossPlotter(mylog_path=log_path, mylog_name="training.log",
                         myloss_names=my_loss, mymetric_names=my_metric, cmb_plot=plt_sep)

    def train(nb_ep, loader_lb, loader_un):
        lss_seg = lss_adv = lss_lab = lss_dis = lsd_lab = lsd_unl = dic_all = acc_all = mn_dac = 0
        plt_lab = plt_unl = plt_lab_gt = plt_unl_gt = 0

        for i, (data_lab, data_unl) in tqdm(enumerate(zip(loader_lb, loader_un))):

            # Train Segmentator
            for param in model_dis.parameters():
                param.requires_grad = False

            for param in model_seg.parameters():
                param.requires_grad = True
                model_seg.zero_grad()
                model_seg.train()

            data_lab.to(device)
            data_unl.to(device)

            optimizer_seg_wh.zero_grad()
            optimizer_seg_mu.zero_grad()
            optimizer_seg_si.zero_grad()

            pred_lab = model_seg(data_lab)
            loss_seg_lab = compute_loss(pred_lab, data_lab.gt, wht)
            dic = compute_dice_metrics(pred_lab, data_lab.gt)
            acc = compute_acc_metrics(pred_lab, data_lab.gt)

            pred_unl = model_seg(data_unl)
            data_unl.x = func.softmax(pred_unl, 1)
            Dis_out_adv = model_dis(data_unl)
            loss_adv_target = bce_loss(Dis_out_adv, source_label)

            # loss_seg = (alpha * loss_seg_lab) + ((1 - alpha) * (lda * loss_adv_target))
            loss_seg = (loss_seg_lab) + ((lda * loss_adv_target))
            loss_seg.backward()

            # Train Discriminator
            for param in model_dis.parameters():
                param.requires_grad = True
                model_dis.zero_grad()
                model_dis.train()

            for param in model_seg.parameters():
                param.requires_grad = False

            pred_lab = pred_lab.detach()
            data_lab.x = func.softmax(pred_lab, 1)
            pred_unl = pred_unl.detach()
            data_unl.x = func.softmax(pred_unl, 1)

            Dis_out_lab = model_dis(data_lab)
            loss_dis_lab = bce_loss(Dis_out_lab, source_label)
            Dis_out_unl = model_dis(data_unl)
            loss_dis_unl = bce_loss(Dis_out_unl, target_label)

            loss_dis = loss_dis_lab + loss_dis_unl
            loss_dis.backward()

            output = (Dis_out_lab > 0.5).float()
            correct = (output == source_label).float().sum() * 50
            mn_dac += correct.item()

            output = (Dis_out_unl > 0.5).float()
            correct = (output == target_label).float().sum() * 50

            mn_dac += correct.item()

            optimizer_seg_wh.step()
            optimizer_seg_mu.step()
            optimizer_seg_si.step()

            optimizer_dis_wh.step()
            optimizer_dis_mu.step()
            optimizer_dis_si.step()

            lss_seg += loss_seg.item()
            lss_adv += loss_adv_target.item()
            lss_lab += loss_seg_lab.item()

            lss_dis += loss_dis.item()
            lsd_lab += loss_dis_lab.item()
            lsd_unl += loss_dis_unl.item()

            dic_all += dic.item()
            acc_all += acc.item()

            plt_lab_gt += torch.histc(torch.max(data_lab.gt, 1)[1], bins=32, min=0, max=31)
            plt_unl_gt += torch.histc(torch.max(data_unl.gt, 1)[1], bins=32, min=0, max=31)
            plt_lab += torch.histc(torch.max(func.softmax(pred_lab, 1), 1)[1], bins=32, min=0, max=31)
            plt_unl += torch.histc(torch.max(func.softmax(pred_unl, 1), 1)[1], bins=32, min=0, max=31)

        metric = np.array(
            [lss_seg / len(train_lb_loader), lss_adv / len(train_lb_loader), lss_lab / len(train_lb_loader),
             lss_dis / len(train_lb_loader), lsd_lab / len(train_lb_loader), lsd_unl / len(train_lb_loader),
             dic_all / len(train_lb_loader), acc_all / len(train_lb_loader), mn_dac / len(train_lb_loader)
             ])
        if plot_hist:
            if not os.path.exists(join(log_path, 'plot', 'hist')):  # Path to save the histogram
                os.makedirs(join(log_path, 'plot', 'hist'))
            plt.figure()
            plt.plot(plt_lab_gt.cpu().numpy(), label="GT_lab")
            plt.plot(plt_unl_gt.cpu().numpy(), label="GT_unl")
            plt.plot(plt_lab.cpu().numpy(), label="Pr_lab")
            plt.plot(plt_unl.cpu().numpy(), label="Pr_unl")
            plt.title("Epcoh_" + str(nb_ep))
            plt.legend()
            plt.savefig(join(log_path, 'plot', 'hist', "Epcoh_" + str(nb_ep) + '.png'))
            plt.close

        # if ((lda * lss_adv / lss_lab).item() > 1) | ((lda * lss_adv / lss_lab).item() < 0.75):
        # if nb_ep < 10:
        #     lda.data = torch.FloatTensor([2 * lss_lab / lss_adv]).to(device)
        # if nb_ep > 10 & nb_ep < 100:
        #     lda.data = torch.FloatTensor([1.5 * lss_lab / lss_adv]).to(device)
        # if nb_ep > 100 & nb_ep < 200:
        #     lda.data = torch.FloatTensor([1.25 * lss_lab / lss_adv]).to(device)
        # if nb_ep > 200:
        #     lda.data = torch.FloatTensor([1 * lss_lab / lss_adv]).to(device)

        # if (nb_ep + 1 % 10) <= 4:
        #     alpha.data = torch.FloatTensor([1.0]).to(device)
        # else:
        #     alpha.data = torch.FloatTensor([0.0]).to(device)
        #
        # if nb_ep > 0:
        #     lda.data = torch.FloatTensor([1 * lss_lab / (lss_adv + 0.000001)]).to(device)
        #
        # if (nb_ep + 1 % 10) == 0:
        #     alpha.data = torch.FloatTensor([0.95 * alpha]).to(device)
        #     if (nb_ep + 1 % 100) == 0:
        #         alpha.data = torch.FloatTensor([1]).to(device)
        # if ((lda * lss_adv / lss_lab).item() > 1.15) | ((lda * lss_adv / lss_lab).item() < 0.85):
        #     lda.data = torch.FloatTensor([1 * lss_lab / lss_adv]).to(device)


        return metric

    def test(loader):
        lss_seg = lss_lab = lss_adv = lss_dis = lsd_lab = lsd_unl = dic_all = acc_all = mn_dac = 0
        dic_arr = []
        acc_arr = []
        model_seg.eval()
        model_dis.eval()

        with torch.no_grad():
            cnt = 0
            for data in tqdm(loader):
                data.to(device)

                file_name = loader.dataset.files[cnt][36:-6]

                spe_inp = data.x
                xyz = data.xyz
                gt_oh = data.gt
                gt = torch.max(data.gt, 1)[1]

                pred_lab = model_seg(data)
                pred_prob = func.softmax(pred_lab, dim=1)
                pred_labels = torch.max(func.softmax(pred_lab, dim=1), 1)[1]

                sio.savemat('output_dm6/' + file_name + '.mat', {'spe_inp': spe_inp.cpu().numpy(),
                                                             'xyz': xyz.cpu().numpy(),
                                                             'gt_oh': gt_oh.cpu().numpy(),
                                                             'gt': gt.cpu().numpy(),
                                                             'pred_prob_out': pred_prob.cpu().numpy(),
                                                             'pred_labels': pred_labels.cpu().numpy()})

                loss_seg_lab = compute_loss(pred_lab, data.gt, wht)
                dic = compute_dice_metrics(pred_lab, data.gt)
                acc = compute_acc_metrics(pred_lab, data.gt)

                lss_lab += loss_seg_lab.item()
                dic_all += dic.item()
                acc_all += acc.item()
                dic_arr.append(dic.item())
                acc_arr.append(acc.item())

                cnt += 1

            metric = np.array([lss_seg / len(loader), lss_adv / len(loader), lss_lab / len(loader),
                               lss_dis / len(loader), lsd_lab / len(loader), lsd_unl / len(loader),
                               dic_all / len(loader), acc_all / len(loader), mn_dac / len(loader),
                               dic_arr, acc_arr])
        return metric

    print("===> Starting Model Training at Epoch: {}".format(initial_epoch))

    for epoch in range(initial_epoch, num_epochs):
        start = time.time()

        print("\n\n")
        print("Epoch:{}".format(epoch))

        test_un_metric = test(test_un_loader)
        print(
            "===> Testing_d2 Epoch {}: Loss = {:.4f}, Mean_Dice_Accuracy = {:.4f}, Std_Dice_Accuracy = {:.4f}, Mean_Accuracy = {:.4f}, Std_Accuracy = {:.4f}".format(
                epoch,
                test_un_metric[2],
                np.mean(test_un_metric[-2]),
                np.std(test_un_metric[-2]),
                np.mean(test_un_metric[-1]),
                np.std(test_un_metric[-1])))

        end = time.time()
        # pdb.set_trace()
        # sio.savemat('bs_dic_dm0.mat', {'dice': np.array(test_un_metric[-2])})

        end = time.time()
        print("===> Epoch:{} Completed in {:.4f} seconds".format(epoch, end - start))

    print("===> Done Training for Total {:.4f} Epochs".format(num_epochs))


if __name__ == "__main__":
    main(_get_config())
