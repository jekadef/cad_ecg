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
parser.add_argument('--version', type=str, required=True)
args = parser.parse_args()
data_set = args.dataset
model_type = args.model
version_date = args.version

# files
data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'
input_fn = 'input_data/cohort_' + data_set + '_input_dict_' + version + '.pkl.gz'
search_fn = 'gridsearch_results/' + model_type + '_' + data_set + '_hyperparam_search_' + version_date + '.pkl.gz'
train_fn = 'bestmodel_results/' + model_type + '_' + data_set + '_model_performance_' + version_date + '.pkl.gz'

with gzip.open(input_fn, 'rb') as f:
    input_dict = pkl.load(f)
data_dict = input_dict['data_dict']
demog_dict = input_dict['demog_dict']

if model_type == 'logistic_regression':
    with gzip.open(search_fn, 'rb') as f:
        lr_search_results = pkl.load(f)
    lr_results = helper.get_lr_model(data_dict, lr_search_results)
    for seed in range(100):
        Xval, yval = resample(data_dict['Xval'],
                              data_dict['yval'],
                              random_state=8,
                              stratify=data_dict['yval'])
        # Logistic Regression Performance Metrics
        logreg_pred = lr_results['model'].predict(Xval)
        lr_results[seed] = helper.get_metrics(yval, logreg_pred)
    with gzip.open(train_fn, 'wb') as f:
        pkl.dump(lr_results, f)

if model_type == 'random_forest':
    with gzip.open(search_fn, 'rb') as f:
        rf_search_results = pkl.load(f)
    rf_results = helper.get_rf_model(data_dict, rf_search_results)
    for seed in range(100):
        Xval, yval = resample(data_dict['Xval'],
                              data_dict['yval'],
                              random_state=8,
                              stratify=data_dict['yval'])
        # Random Forest Performance Metrics
        ranfor_pred = rf_results['model'].predict(Xval)
        rf_results[seed] = helper.get_metrics(yval, ranfor_pred) # save metrics in dict
    with gzip.open(train_fn, 'wb') as f:
        pkl.dump(rf_results, f)
