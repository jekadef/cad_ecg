import sys
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import gzip

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--trainset', type=str, required=True)
parser.add_argument('--testset', type=str, required=True)
parser.add_argument('--version', type=str, required=True)
args = parser.parse_args()
model_type = args.model
version_date = args.version
train_set = args.trainset
test_set = args.testset

# train_set = ['t_plus_1', 't_minus_1', 't_minus_5', 't_minus_10', 't_plus_1', 't_plus_1', 't_plus_1']
# test_set = ['t_plus_1', 't_minus_1', 't_minus_5', 't_minus_10', 't_minus_1', 't_minus_5', 't_minus_10']

results_dict = {}
data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'

input_fn = data_dir + 'cohort_' + test_set + '_input_dict_' + version_date + '.pkl.gz'
train_fn = dat_dir + 'bestmodel_results/' + model_type + '_' + train_set + '_model_performance_' + version_date + '.pkl.gz'
results_fn = 'bestmodel_results/' + model_type + '_performance_' + train_set + '_evaluated_on_' + test_set + ' ' + version_date + '.pkl.gz'
with gzip.open(train_fn, 'rb') as f:
    trained_results = pkl.load(f)
# feat_importance = trained_results['model'].feature_importances_

with gzip.open(input_fn, 'rb') as f:
    test_input_dict = pkl.load(f)
test_data_dict = test_input_dict['data_dict']
test_demog_dict = test_input_dict['demog_dict']
mod_results = pd.DataFrame()
test_results = pd.DataFrame()
for seed in range(10):
    Xtest, ytest = resample(test_data_dict['Xtest'],
                            test_data_dict['ytest'],
                            random_state=seed,
                            stratify=test_data_dict['ytest'])
    # Logistic Regression Performance Metrics
    mod_pred = trained_results['model'].predict(Xtest)
    mod_metrics = helper.get_metrics(ytest, mod_pred)
    # lr_results = pd.concat([lr_results, pd.Series(logreg_metrics['f1-score'])])
    mod_results = pd.concat([mod_results, pd.DataFrame.from_dict(mod_metrics, orient='index')], axis=1)
test_results['test'] = round(pd.DataFrame(mod_results.mean(axis=1)), 3)

mod_results = pd.DataFrame()
val_results = pd.DataFrame()
for seed in range(10):
    Xval, yval = resample(test_data_dict['Xval'],
                          test_data_dict['yval'],
                          random_state=seed,
                          stratify=test_data_dict['yval'])
    # Logistic Regression Performance Metrics
    mod_pred = trained_results['model'].predict(Xval)
    mod_metrics = helper.get_metrics(yval, mod_pred)
    # lr_results = pd.concat([lr_results, pd.Series(logreg_metrics['f1-score'])])
    mod_results = pd.concat([mod_results, pd.DataFrame.from_dict(mod_metrics, orient='index')], axis=1)
val_results['val'] = round(pd.DataFrame(mod_results.mean(axis=1)), 3)
results_dict[str(train_set) + '_eval_on_' + str(test_set)] = pd.concat([test_results, val_results], axis=1)

print(results_dict)
