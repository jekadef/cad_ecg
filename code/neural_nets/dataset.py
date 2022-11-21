from torch.utils.data import Dataset, DataLoader, Subset
import sys
sys.path.insert(0, '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/code/neural_nets')
from sampler import BalancedBatchSampler
import gzip
import pandas as pd
import numpy as np
import base64
import xmltodict
from scipy import signal
import pickle as pkl
import sys
sys.path.insert(0, '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/code/neural_nets')


def load_ecg(filen):
    with open(filen, 'r') as infile:
        try:
            ecg_data = xmltodict.parse(infile.read())
            ecg_data = ecg_data['RestingECG']
        except:  # Blank excepts aren't recommended
            return False
            # specify which error
            print("can't load ecg")
    return ecg_data


class ECGDataset(Dataset):
    def __init__(self, group, data_format, data_set):
        self.group = group # what is this group thing??? Ah train, test, val, temp
        self.data_format = data_format
        self.dataset = data_set
        data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
        fn = data_dir + 'cohort_' + data_set + '_input_dict_20221102.pkl.gz'

        with gzip.open(fn, 'rb') as f:
            input_dict = pkl.load(f)

        self.ecg_paths = input_dict['demog_dict'][group]['full_path']
        self.ecg_labels = input_dict['demog_dict'][group]['label'] # GET THIS to pd not np
        self.ecg_labels = self.ecg_labels.astype({'label': 'int32'})
        self.n_samples = len(self.ecg_labels) # is this right??

    def __getitem__(self, idx):
        path_key = self.ecg_paths.iloc[idx]
        data = load_ecg(path_key)
        raw_lead_data = data['Waveform'][1]['LeadData']
        leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        ecg_waveform = []
        ecg_spec = []
        for this_lead in raw_lead_data:
            lead_name = this_lead['LeadID']
            waveform = base64.b64decode(this_lead['WaveFormData'])  # Base64 strings. This returns a bytes object - Which is 16bit integer in turn
            waveform = np.frombuffer(waveform, dtype='int16')
            if len(waveform) != 5000:
                waveform = np.concatenate((waveform, waveform), axis=None)
            # bandpass and median filters
            try:
                sig = signal.medfilt2d(waveform)
                sos = signal.butter(3, np.array([0.5, 40]), btype='bandpass', output='sos', fs=500)
                filtered = signal.sosfilt(sos, sig)
            except:
                print('filters do not work')
            if lead_name not in leads:
                break
            if self.data_format == 'waveform':
                ecg_waveform.append(filtered)
            if self.data_format == 'spectrogram':
                f, t, ecg_spec = signal.spectrogram(filtered, return_onesided=False, noverlap=0, fs=500)
                ecg_waveform.append(ecg_spec)
        if len(ecg_waveform) == 8:
            # add it to the df # if not then move on
            ecg = np.array(ecg_waveform)
            label = self.ecg_labels.iloc[idx]

        return ecg, label, path_key

    def __len__(self):
        return self.n_samples


# class ECGSpectrogramDataset(Dataset):
#     def __init__(self, group, timeframe, windowsize):
#         self.group = group # what is this group thing??? Ah train, test, val, temp
#         data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/cohort_data/'
#         fn = data_dir + 'cohort_' + self.cli_args.dataset + '_input_dict_20220912.pkl.gz'
#
#         with gzip.open(fn, 'rb') as f:
#             input_dict = pkl.load(f)
#
#         self.ecg_paths = input_dict['demog_dict'][group]['full_path']
#         self.ecg_labels = input_dict['demog_dict'][group]['label'] # GET THIS to pd not np
#         self.ecg_labels = self.ecg_labels.astype({'label': 'int32'})
#         self.n_samples = len(self.ecg_labels) # is this right??
#
#     def __getitem__(self, idx):
#         path_key = self.ecg_paths.iloc[idx]
#         data = load_ecg(path_key)
#         raw_lead_data = data['Waveform'][1]['LeadData']
#         leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#         ecg_waveform = []
#         ecg_spec = []
#         for this_lead in raw_lead_data:
#             lead_name = this_lead['LeadID']
#             waveform = base64.b64decode(this_lead['WaveFormData'])  # Base64 strings. This returns a bytes object - Which is 16bit integer in turn
#             waveform = np.frombuffer(waveform, dtype='int16')
#             #waveform = waveform[0:2500]
#             # print(len(waveform))
#             if len(waveform) != 5000:
#                 waveform = np.concatenate((waveform, waveform), axis=None)
#             # bandpass and median filters
#             try:
#                 sig = signal.medfilt2d(waveform)
#                 sos = signal.butter(3, np.array([0.5, 40]), btype='bandpass', output='sos', fs=500)
#                 filtered = signal.sosfilt(sos, sig)
#             except:
#                 print('filters do not work')
#             if lead_name not in leads:
#                 break
#             f, t, ecg_spec = signal.spectrogram(filtered, return_onesided=False, noverlap=0, fs=500)
#             ecg_waveform.append(ecg_spec)
#         if len(ecg_waveform) == 8:
#             # add it to the df # if not then move on
#             ecg = np.array(ecg_waveform)
#             # print(ecg.shape)
#             label = self.ecg_labels.iloc[idx]
#         return ecg, label, path_key
#
#     def __len__(self):
#         return self.n_samples
#



#
# class MyIterableDataset(torch.utils.data.IterableDataset):
#     def __init__(self, start, end):
#         super(MyIterableDataset).__init__()
#         assert end > start, "this example code only works with end >= start"
#         self.start = start
#         self.end = end
#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info is None:  # single-process data loading, return the full iterator
#             iter_start = self.start
#             iter_end = self.end
#         else:  # in a worker process
#             # split workload
#             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
#             worker_id = worker_info.id
#             iter_start = self.start + worker_id * per_worker
#             iter_end = min(iter_start + per_worker, self.end)
#         return iter(range(iter_start, iter_end))
# # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
# ds = MyIterableDataset(start=3, end=7)
#
# # Single-process loading
# print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
#
# # Mult-process loading with two worker processes
# # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
# print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
#
# # With even more workers
# print(list(torch.utils.data.DataLoader(ds, num_workers=20)))