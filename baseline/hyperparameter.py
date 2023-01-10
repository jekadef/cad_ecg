import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import gzip
import sys
import helper

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--searchtype', type=str, required=True)
parser.add_argument('--version', type=str, required=True)
args = parser.parse_args()
model_type = args.model
data_set = args.dataset
search_type = args.searchtype
version_date = args.version

# files
data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'
input_fn = 'input_data/cohort_' + data_set + '_input_dict_' + version + '.pkl.gz'
result_fn = 'gridsearch_results/' + model_type + '_' + data_set + '_grid_results_' + version + '.pkl.gz'

with gzip.open(input_fn, 'rb') as f:
    input_dict = pkl.load(f)
data_dict = input_dict['data_dict']
demog_dict = input_dict['demog_dict']
# uh oh predefined split in tminus... does that exist?

if model_type == 'logistic_regression':
    lr_search_results = helper.lr_search(search_type, data_dict, utils.lr_grid)
    with gzip.open(result_fn, 'wb') as f:
        pkl.dump(lr_search_results, f)

elif model_type == 'random_forest':
    rf_search_results = helper.rf_search(search_type, data_dict, utils.rf_grid)
    with gzip.open(result_fn, 'wb') as f:
        pkl.dump(rf_search_results, f)

