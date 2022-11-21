import argparse
import datetime
import os
import struct
import sys
import hashlib
import shutil
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics

sys.path.insert(0, '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/code/neural_nets')
from util import enumerate_estimated, EarlyStopping, LRScheduler
from dataset import ECGDataset
from logconf import logging
# import model
from model import EcgNet, EcgNetD, EcgNetE, EcgNetF, EcgNetG, EcgNetH, EcgNetI, EcgNetJ, EcgNetK, EcgNetL, EcgNetM
from model import EcgNetN, EcgNetO, EcgNetP, EcgNetQ, EcgNetR, EcgNetS, EcgNetT, EcgNetU, EcgNetV, EcgNetW, EcgNetX
from model import EcgNetY, EcgNetAA, EcgNetBB, EcgNetCC, EcgNetDD, EcgNetEE, EcgNetFF, EcgNetGG, EcgNetHH, EcgNetII
from model import EcgNetJJ, EcgNetKK, EcgNetLL, EcgNetMM, EcgNetNN, EcgNetOO, EcgNetPP, EcgNetQQ, EcgNetRR, EcgNetSS
from model import EcgNetTT, EcgNetUU, EcgNetVV, EcgNetWW, EcgNetXX, EcgNetYY, EcgNetZZ, EcgNetAAA, EcgNetBBB, EcgNetCCC
from model import EcgNetDDD, EcgNetEEE, EcgNetFFF, EcgNetGGG, EcgNetHHH, EcgNetIII, EcgNetJJJ, EcgNetKKK

from unet_model import UNet
#torch.cuda.empty_cache()

from matplotlib import pyplot
import torch.nn.functional as F
import base64
import xmltodict
import gzip
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from scipy import signal

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# CUDA_LAUNCH_BLOCKING = 1
# NUMEXPR_MAX_THREADS = ?

# Used for compute_batch_loss and log_metrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_PRED_P_NDX = 2
METRICS_LOSS_NDX = 3
METRICS_PATH_KEY = 4
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9
METRICS_SIZE = 10


class EcgTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                            help="Number of worker processes for background data loading",
                            default=1,
                            type=int,
                            )
        parser.add_argument('--batch-size',
                            help="Batch size to use for training",
                            default=32,
                            )
        parser.add_argument('--tb-prefix',
                            default='experimentX',
                            help="Data prefix to use for Tensorboard run. Defaults to experiment",
                            )
        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='cad_ecg',
                            )
        parser.add_argument('--trained_model',
                            type=str,
                            help='name model'
                            )
        parser.add_argument('--dataset',
                            type=str,
                            help='which dataset to train the CNN on')
        parser.add_argument('--data-type',
                            type=str,
                            help='processed waveform or spectrogram data to input in the model'
                            )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.val_writer = None

        if torch.cuda.is_available():
            print('We have a GPU!')
        else:
            print('only CPU >;(')
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.pretrain_model = self.model_load()
        self.pred_path_df = pd.DataFrame()

    def initiate_val_dl(self, group_set):
        """
        Initiate the validation set dataloader
        Returns:
            val_dl (DataLoader object):
        """

        data_type = self.cli_args.data_type
        batch_size = self.cli_args.batch_size
        dataset = self.cli_args.dataset

        val_ds = ECGDataset(group=group_set, data_format=data_type, data_set=dataset)

        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(val_ds, batch_size=int(batch_size), num_workers=self.cli_args.num_workers,
                            pin_memory=self.use_cuda, shuffle=False)

        return val_dl

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g, classification_threshold=0.5):
        """
        Compute the Mean Cross Entropy Loss of a batch

        Args:
            batch_ndx (): the batch index
            batch_tup(): the batch
            batch_size(): the number of samples in the batch
            metrics_g(): the object containing the metric from the batch

        Returns:
            mean loss of batch
        """
        experiment = self.cli_args.tb_prefix
        input_t, label_t, path_ecg_key = batch_tup

        data_type = self.cli_args.data_type

        experiment = self.cli_args.tb_prefix
        expr_str = str.split(experiment, '_')
        expr = expr_str[1]

        waveform_1D = ['F','G','H','I','N','O','R','S','T','U','X','Y','EE','FF','HH','II','JJ', 'AAA', 'BBB']
        waveform_2D = ['D','E','L','M','V','W','GG', 'KK', 'LL', 'MM', 'NN', 'OO', 'PP']

        spectrogram_2D = ['J','K','P','Q','AA','BB', 'QQ', 'RR', 'SS', 'TT', 'ZZ']
        spectrogram_3D = ['CC','DD', 'UU', 'VV', 'WW', 'XX', 'YY']

        if data_type == 'waveform':
            if expr in waveform_1D:
                input_t = input_t.view(-1, 8, 5000)
                # print('a')
            if expr in waveform_2D:
                input_t = input_t.view(-1, 1, 8, 5000)
                # print('b')

        if data_type == 'spectrogram':
            if expr in spectrogram_2D:
               input_t = input_t.view(-1, 8, 19, 256)
                # print('c')
            if expr in spectrogram_3D:
                input_t = input_t.view(-1, 1, 8, 19, 256)
                # print('d')

        input_g = input_t.to(self.device,
                             dtype=torch.float32,
                             non_blocking=True
                             )
        label_g = label_t.to(self.device,
                             non_blocking=True,
                             dtype=torch.long,
                             )

        logits_g, probability_g = self.pretrain_model(input_g)

        # why logits this shape??
        # per sample loss
        loss_func = nn.CrossEntropyLoss(reduction='none')
        #loss_g = loss_func(logits_g, label_g[:,1],)
        loss_g = loss_func(logits_g, label_g, )

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        ### added lines
        with torch.no_grad():
            prediction_bool_g = (logits_g[:, 0:1] > classification_threshold).to(torch.float32)

            tp = (prediction_bool_g * label_g).sum()
            fn = ((1 - prediction_bool_g) * label_g).sum()
            fp = (prediction_bool_g * (~label_g)).sum()

            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp  # something wrong with false positive

            metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g.detach()
            metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

            pred_pt = prediction_bool_g.detach()
            pred_pt = pred_pt.cpu()
            pred_pt = pred_pt.squeeze()

            df = pd.concat([pd.Series(pred_pt.numpy()), pd.Series(path_ecg_key)], axis=1, ignore_index=True)

            self.pred_path_df = pd.concat([self.pred_path_df, df], axis=0, ignore_index=True)

        # recombine loss per sample into loss per batch
        return loss_g.mean()

    def do_validation(self, val_dl):
        """
        Perform a validation loop
        Args:
            epoch_ndx (int): the index of the epoch
            val_dl (DataLoader): validation set dataloader

        Returns:
            valMetrics_g (tensor)
        """
        with torch.no_grad():
            self.pretrain_model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerate_estimated(
                val_dl,
                "val",
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup, in batch_iter:
                self.compute_batch_loss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g,
                )

        # with torch.no_grad():
        #     self.pretrain_model.eval()
        #     valMetrics_g = torch.zeros(
        #         METRICS_SIZE,
        #         len(val_dl.dataset),
        #         device=self.device,
        #     )
        #
        #     self.compute_loss(val_dl, valMetrics_g)

        return valMetrics_g.to('cpu')

    def model_load(self):
        """
        Load the pretrained model as specified by 'trained_model' variable
        """
        experiment = self.cli_args.tb_prefix
        model_file = self.cli_args.trained_model

        if experiment == 'expr_DD':
            model = EcgNetDD()
        elif experiment == 'expr_EE':
            model = EcgNetEE()
        elif experiment == 'expr_FF':
            model = EcgNetFF()
        elif experiment == 'expr_II':
            model = EcgNetII()
        elif experiment == 'expr_JJ':
            model = EcgNetJJ()
        elif experiment == 'expr_KK':
            model = EcgNetKK()
        elif experiment == 'expr_LL':
            model = EcgNetLL()
        elif experiment == 'expr_MM':
            model = EcgNetMM()
        elif experiment == 'expr_NN':
            model = EcgNetNN()
        elif experiment == 'expr_OO':
            model = EcgNetOO()
        elif experiment == 'expr_PP':
            model = EcgNetPP()
        elif experiment == 'expr_QQ':
            model = EcgNetQQ()
        elif experiment == 'expr_RR':
            model = EcgNetRR()
        elif experiment == 'expr_SS':
            model = EcgNetSS()
        elif experiment == 'expr_TT':
            model = EcgNetTT()
        elif experiment == 'expr_UU':
            model = EcgNetUU()
        elif experiment == 'expr_VV':
            model = EcgNetVV()
        elif experiment == 'expr_WW':
            model = EcgNetWW()
        elif experiment == 'expr_XX':
            model = EcgNetXX()
        elif experiment == 'expr_YY':
            model = EcgNetYY()
        elif experiment == 'expr_ZZ':
            model = EcgNetZZ()
        elif experiment == 'expr_AAA':
            model = EcgNetAAA()
        elif experiment == 'expr_BBB':
            model = EcgNetBBB()
        elif experiment == 'expr_CCC':
            model = EcgNetCCC()
        elif experiment == 'expr_DDD':
            model = EcgNetDDD()
        elif experiment == 'expr_EEE':
            model = EcgNetEEE()
        elif experiment == 'expr_FFF':
            model = EcgNetFFF()
        elif experiment == 'expr_GGG':
            model = EcgNetGGG()
        elif experiment == 'expr_HHH':
            model = EcgNetHHH()
        elif experiment == 'expr_III':
            model = EcgNetIII()
        elif experiment == 'expr_JJJ':
            model = EcgNetJJJ()
        elif experiment == 'expr_KKK':
            model = EcgNetKKK()

        #     state = {
        #         'sys_argv': sys.argv,
        #         'time': str(datetime.datetime.now()),
        #         'model_state': model.state_dict(),
        #         'model_name': type(model).__name__,
        #         'optimizer_state': self.optimizer.state_dict(),
        #         'optimizer_name': type(self.optimizer).__name__,
        #         'epoch': epoch_ndx,
        #         'totalTrainingSamples_count': self.totalTrainingSamples_count,
        #     }

        file_path = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/models/from20221102/' + str(model_file)
        state = torch.load(file_path)

        model.load_state_dict(state['model_state'])
        self.totalTrainingSamples_count = state['totalTrainingSamples_count']
        # model = model.to(self.device)

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return model

    def initiate_tensorboard_writers(self):
        """
        Initiate the tensorboard writer and specify how the performance metrics are to be written

        """
        log_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/neuralnet_result/from20221102/' + \
                  str(self.cli_args.tb_prefix) + '/'
        self.val_writer = SummaryWriter(log_dir=log_dir + self.time_str +
                                                '_evaluate-after_experiment-' + str(self.cli_args.tb_prefix) +
                                                '_val_' + str(self.cli_args.data_type) + '_' + str(self.cli_args.dataset))

        self.test_writer = SummaryWriter(log_dir=log_dir + self.time_str +
                                                '_evaluate-after_experiment-' + str(self.cli_args.tb_prefix) +
                                                '_test_' + str(self.cli_args.data_type) + '_' + str(self.cli_args.dataset))

    def log_metrics(self, mode_str, metrics_t, classificationThreshold=0.5):
        """
        Write out the performance metrics to the commandline and to the tensorboard
        Args:
            epoch_ndx (int): index of the epoch
            mode_str (str): whether in training or validation mode
            metrics_t (idk): the performance metrics
            classificationThreshold (float): the value to partition the binary classification
        """
        self.initiate_tensorboard_writers()

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        all_label_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        pos = 'pos'
        neg = 'neg'

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum())
        truePos_count = pos_correct = int((posLabel_mask & posPred_mask).sum())

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        metrics_dict['percent_all/tp'] = sum_a[METRICS_TP_NDX] / (all_label_count or 1) * 100
        metrics_dict['percent_all/fn'] = sum_a[METRICS_FN_NDX] / (all_label_count or 1) * 100
        metrics_dict['percent_all/fp'] = sum_a[METRICS_FP_NDX] / (all_label_count or 1) * 100

        precision = metrics_dict['pr/precision'] = truePos_count / np.float32(truePos_count + falsePos_count)
        recall = metrics_dict['pr/recall'] = truePos_count / np.float32(truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)

        # fpr, tpr, thresholds = metrics.roc_curve(metrics_t[METRICS_LABEL_NDX], metrics_t[METRICS_PRED_NDX], pos_label=1)
        # auc = metrics.auc(fpr, tpr)
        # metrics_dict['auc'] = auc
        #
        # aucpr = metrics.average_precision_score(metrics_t[METRICS_LABEL_NDX], metrics_t[METRICS_PRED_NDX], pos_label=1)
        # metrics_dict['aucpr'] = aucpr

        # fig = pyplot.figure()
        # pyplot.plot(fpr, tpr)
        # writer.add_figure('roc', fig, self.totalTrainingSamples_count)

        # writer.add_scalar('auc', auc, self.totalTrainingSamples_count)

        # bins = np.linspace(0, 1)

        # writer.add_histogram(
        #    'label_neg',
        #    metrics_t[METRICS_PRED_P_NDX, negLabel_mask],
        #    self.totalTrainingSamples_count,
        #    bins=bins
        # )
        # writer.add_histogram(
        #    'label_pos',
        #    metrics_t[METRICS_PRED_P_NDX, posLabel_mask],
        #    self.totalTrainingSamples_count,
        #    bins=bins
        # )

        writer = getattr(self, mode_str + '_writer')

        # for key, value in metrics_dict.items():
        #    writer.add_scalar(key, value, self.totalTrainingSamples_count)

        for key, value in metrics_dict.items():
            # key = key.replace('pos', pos)
            # key = key.replace('neg', neg)
            # total training samples count is the X-axis
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.flush()

        score = metrics_dict['pr/f1_score']
        return score

    def main(self):
        """
        Perform the validation of the pretrained model.
        """

        data_type = self.cli_args.data_type
        experiment = self.cli_args.tb_prefix
        expr_str = str.split(experiment, '_')
        expr = expr_str[1]

        val_dl = self.initiate_val_dl('val')
        test_dl = self.initiate_val_dl('test')

        valMetrics_t = self.do_validation(val_dl)
        # val_loss = valMetrics_t[METRICS_LOSS_NDX].mean()

        testMetrics_t = self.do_validation(test_dl)
        # test_loss

        self.log_metrics('val', valMetrics_t)
        self.log_metrics('test', testMetrics_t)

        # # save valMetrics
        # filename = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/metrics_evaluate-pretrained_experiment-' \
        #            + str(self.cli_args.tb_prefix) + '_' + str(self.cli_args.data_type) + '_' + str(self.cli_args.dataset) + '.pkl.gz'
        #
        # with gzip.open(filename, 'wb') as f:
        #     pkl.dump(self.pred_path_df, f)
        # # torch.save(valMetrics_t, filename)

if __name__ == '__main__':
    EcgTrainingApp().main()

