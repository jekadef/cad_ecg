import argparse
import datetime
import os
import struct
import sys
import hashlib
import shutil
import numpy as np
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
import pandas as pd
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
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=10,
                            type=int,
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
        parser.add_argument('--lr-scheduler',
                            dest='lr_scheduler',
                            action='store_true',
                            )
        parser.add_argument('--early-stopping',
                            dest='early_stopping',
                            action='store_true',
                            )
        parser.add_argument('--pretrained_model',
                            type=str,
                            help='which model to continue training on'
                            )
        parser.add_argument('--continue-training',
                            action='store_true',
                            dest='continue_training'
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

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        if torch.cuda.is_available():
            print('We have a GPU!')
        else:
            print('only CPU >;(')
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self.initiate_model()
        self.optimizer = self.initiate_optimizer()
        
        if self.cli_args.lr_scheduler: 
            self.scheduler = self.initiate_scheduler()
            
        if self.cli_args.early_stopping:
            self.earlystop = EarlyStopping()

        if self.cli_args.continue_training:
            self.model = self.initiate_pretrained_model(self.cli_args.pretrained_model)

    def initiate_train_dl(self):

        """
        Initiate the training set dataloader.
        Returns:
            train_dl (DataLoader object)

        """
        data_type = self.cli_args.data_type
        batch_size = self.cli_args.batch_size
        dataset = self.cli_args.dataset

        train_ds = ECGDataset(group="train", data_format=data_type, data_set=dataset)

        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(train_ds, batch_size=int(batch_size), num_workers=self.cli_args.num_workers,
                              pin_memory=self.use_cuda, shuffle=True)
        return train_dl

    def initiate_val_dl(self):
        """
        Initiate the validation set dataloader
        Returns:
            val_dl (DataLoader object):
        """
        data_type = self.cli_args.data_type
        batch_size = self.cli_args.batch_size
        dataset = self.cli_args.dataset

        val_ds = ECGDataset(group="val", data_format=data_type, data_set=dataset)

        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(val_ds, batch_size=int(batch_size), num_workers=self.cli_args.num_workers,
                            pin_memory=self.use_cuda, shuffle=False)
        return val_dl

    def initiate_model(self):
        """
        Initiate the convolutional neural network as defined in model.py script.
        Returns:
            model object

        """
        experiment = self.cli_args.tb_prefix

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

        if self.use_cuda:
            log.info('Using CUDA; {} devices'.format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return model

    def initiate_pretrained_model(self, model_file):
        """
        Initiate the convolutional neural network as defined in model.py script.
        Returns:
            model object

        """
        experiment = self.cli_args.tb_prefix

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


        file_path = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/models/from20221102/' + str(model_file)
        state = torch.load(file_path)

        model.load_state_dict(state['model_state'])
        self.totalTrainingSamples_count = state['totalTrainingSamples_count']

        if self.use_cuda:
            log.info('Using CUDA; {} devices'.format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)

        return model

    def initiate_optimizer(self):
        """
        Initiate the optimizer.
        Returns:

        """
        # return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        return optim.Adam(self.model.parameters(), lr=1e-3)

    def initiate_scheduler(self):
        """
        Initiate the learning rate scheduler.
        Returns:

        """
        return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=8)
    
    def main(self):
        """
        Perform the training and validation of the model.
        """
        log.info('Starting {}, {}'.format(type(self).__name__, self.cli_args))

        train_dl = self.initiate_train_dl()
        val_dl = self.initiate_val_dl()

        best_score = 0.0
        current_lr = 1e-3
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))
            
            if current_lr != self.scheduler.optimizer.param_groups[0]['lr']:
                print(f"INFO: Learning rate has decreased to {self.scheduler.optimizer.param_groups[0]['lr']}")
            current_lr = self.scheduler.optimizer.param_groups[0]['lr']
            
            trnMetrics_t = self.do_training(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, 'trn', trnMetrics_t, current_lr)

            valMetrics_t = self.do_validation(epoch_ndx, val_dl)
            score = self.log_metrics(epoch_ndx, 'val', valMetrics_t, current_lr)
            val_loss = valMetrics_t[METRICS_LOSS_NDX].mean()
            best_score = max(score, best_score)
            
            self.save_model('cad', epoch_ndx, score == best_score)

            #### if using learning rate scheduler, increment the scheduler at end of each epoch
            if self.cli_args.lr_scheduler:
                self.scheduler.step(val_loss)
            
            #### if using early stopping and have reached lowest learning rate, increment patience and break if needed 
            if self.cli_args.early_stopping:
                if current_lr == 1e-10:
                    self.earlystop(val_loss)
                    if self.earlystop.early_stop:
                        break

    def do_training(self, epoch_ndx, train_dl):
        """
        Perform a training loop
        Args:
            epoch_ndx (int): the index of the epoch
            train_dl (DataLoader): training set dataloader

        Returns:
            trnMetrics_g (tensor)
        """
        self.model.train()
        # initialize empty metrics array 
        # per class metrics: purpose is info 
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device, )
        # set up batch looping with time estimate
        batch_iter = enumerate_estimated(train_dl, "E{} Training".format(epoch_ndx), start_ndx=train_dl.num_workers, )

        for batch_ndx, batch_tup in batch_iter:
            # free any leftover gradient tensors
            self.optimizer.zero_grad()
            # calculate the loss over a batch of samples
            loss_var = self.compute_batch_loss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            # update model weights
            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')

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

        waveform_1D = ['F','G','H','I','N','O','R','S','T','U','X','Y','EE','FF','HH','II','JJ','AAA','BBB','CCC','DDD','EEE']
        waveform_2D = ['D','E','L','M','V','W','GG', 'KK', 'LL', 'MM', 'NN', 'OO', 'PP', 'FFF', 'GGG', 'HHH', 'III', 'JJJ', 'KKK']

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

        logits_g, probability_g = self.model(input_g)

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

            #metrics_g[METRICS_PATH_KEY, start_ndx:end_ndx] = path_ecg_key

        # recombine loss per sample into loss per batch
        return loss_g.mean()

    def do_validation(self, epoch_ndx, val_dl):
        """
        Perform a validation loop
        Args:
            epoch_ndx (int): the index of the epoch
            val_dl (DataLoader): validation set dataloader

        Returns:
            valMetrics_g (tensor)
        """
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerate_estimated(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup, in batch_iter:
                self.compute_batch_loss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g
                )

        return valMetrics_g.to('cpu')

    def initiate_tensorboard_writers(self):
        """
        Initiate the tensorboard writer and specify how the performance metrics are to be written

        """
        if self.trn_writer is None:
            log_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/neuralnet_result/from20221102/'
            self.trn_writer = SummaryWriter(
                log_dir=log_dir + self.time_str + '-train_classifer_-experiment_' + str(self.cli_args.tb_prefix) +
                        str(self.cli_args.data_type) + str(self.cli_args.dataset))
            self.val_writer = SummaryWriter(
                log_dir=log_dir + self.time_str + '-val_classifer_-experiment_' + str(self.cli_args.tb_prefix) + '_' +
                        str(self.cli_args.data_type) + '_' + str(self.cli_args.dataset))

    def save_model(self, type_str, epoch_ndx, is_best=False):
        """
        Save the model to a file.
        Args:
            type_str ():
            epoch_ndx ():
            is_best ():


        Description of arguments if any
        Description of the return values if any
        Description of errors raised if any
        Optional extra notes or examples of usage
        """
        file_path = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/models/from20221102/' + \
                    str(self.cli_args.tb_prefix) + str(type_str) + str(self.time_str) + str(self.cli_args.tb_prefix) + \
                    '_' + str(self.cli_args.data_type) + '_' + str(self.cli_args.dataset) + '_' + \
                    str(self.totalTrainingSamples_count) + '.state'

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        # get rid of data parallel if it exists
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }

        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        if is_best:
            best_path = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/models/' + str(self.cli_args.tb_prefix) \
                        + '/' + str(type_str) + str(self.time_str) + '_' + '_' \
                        + str(self.totalTrainingSamples_count) + '_best:qq.state'

            # best_path = os.path.join(
            #     'models',
            #     self.cli_args.tb_prefix,
            #     f'{type_str}_{self.time_str}_{self.cli_args.comment}_best-state')
            # shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

    def log_metrics(self, epoch_ndx, mode_str, metrics_t, lr, classificationThreshold=0.5):
        """
        Write out the performance metrics to the commandline and to the tensorboard
        Args:
            epoch_ndx (int): index of the epoch
            mode_str (str): whether in training or validation mode
            metrics_t (idk): the performance metrics
            classificationThreshold (float): the value to partition the binary classification
        """
        self.initiate_tensorboard_writers()

        log.info("E{} {}".format(epoch_ndx, type(self).__name__, ))

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

        metrics_dict['learning_rate'] = lr

        # threshold = torch.linspace(1, 0)
        # tpr = (metrics_t[None, METRICS_PRED_P_NDX, posLabel_mask] >= threshold[:, None]).sum(1).float() / pos_count
        # fpr = (metrics_t[None, METRICS_PRED_P_NDX, negLabel_mask] >= threshold[:, None]).sum(1).float() / neg_count
        # fp_diff = fpr[1:] - fpr[:-1]
        # tp_avg = (tpr[1:] + tpr[:-1]) / 2
        # auc = (fp_diff * tp_avg).sum()

        fpr, tpr, thresholds = metrics.roc_curve(metrics_t[METRICS_LABEL_NDX], metrics_t[METRICS_PRED_NDX], pos_label=1)
        auc = metrics.auc(fpr, tpr)
        metrics_dict['auc'] = auc

        aucpr = metrics.average_precision_score(metrics_t[METRICS_LABEL_NDX], metrics_t[METRICS_PRED_NDX], pos_label=1)
        metrics_dict['aucpr'] = aucpr

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             + "{pr/precision:.4f} precision, "
             + "{pr/recall:.4f} recall, "
             + "{pr/f1_score:.4f} f1 score"
             + "{auc:.4f} auc"
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
             + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
             + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
             ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict
            )
        )

        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
                  ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

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
            key = key.replace('pos', pos)
            key = key.replace('neg', neg)
            # total training samples count is the X-axis
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.flush()

        score = metrics_dict['pr/f1_score']
        return score


if __name__ == '__main__':
    EcgTrainingApp().main()

