import sys
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import numpy as np
import pandas as pd
import pickle as pkl
import gzip

sys.path.insert(0, '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/code/baseline_models')
import utils
import preprocess
import split
import evaluate
import visualization

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--timeframe', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()


    # time_frame = args.timeframe
    data_set = args.dataset
    model_type = args.model

    data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
    fn = data_dir + 'cohort_' + data_set + '_input_dict_20221102.pkl.gz'
    with gzip.open(fn, 'rb') as f:
        input_dict = pkl.load(f)

    data_dict = input_dict['data_dict']
    demog_dict = input_dict['demog_dict']

    if model_type == 'logistic_regression':
        lr_results = {}
        dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'
        fn = dat_dir + 'gridsearch_results/logistic_regression_' + data_set + '_grid_results_20221102.pkl.gz'

        with gzip.open(fn, 'rb') as f:
            lr_search_results = pkl.load(f)

        logreg_bestparams = lr_search_results['best_params']

        if utils.lr_bestmodel is None:
            # Logistic Regression
            # Train model using best hyperparameters
            logreg_model = evaluate.get_model(data_dict['Xtrain'],
                                              data_dict['ytrain'],
                                              'logistic_regression',
                                              logreg_bestparams,
                                              seed=8)

            lr_results['model'] = logreg_model
            lr_results['coefficient'] = logreg_model.coef_
            lr_results['beta'] = logreg_model.intercept_
            # lr_results['features'] = logreg_model.feature_names_in_

            dat_file = 'bestmodel_results/' + model_type + '_' + data_set + '_best_model_performance_20221102.pkl.gz'
            fn = dat_dir + dat_file

            with gzip.open(fn, 'wb') as f:
                pkl.dump(lr_results, f)

        else:
            with gzip.open(fn, 'rb') as f:
                lr_results = pkl.load(f)

        for seed in range(100):
            Xval, yval = resample(data_dict['Xval'],
                                  data_dict['yval'],
                                  random_state=seed,
                                  stratify=data_dict['yval'])

            # Logistic Regression Performance Metrics
            logreg_pred = lr_results['model'].predict(Xval)
            logreg_metrics = evaluate.get_metrics(yval, logreg_pred)
            lr_results[seed] = logreg_metrics

        # roc and prc plots for validation data
        logreg_roc, logreg_prc = visualization.plot_curves(lr_results['model'], data_dict['Xval'], data_dict['yval'])

        pn = dat_dir + 'bestmodel_plots/ROC_plot_' + model_type + '_' + data_set +'_bestmodel_20221102.pdf'
        logreg_roc
        plt.savefig(pn)
        plt.close()

        pn = dat_dir + 'bestmodel_plots/PRC_plot_' + model_type + '_' + data_set +'_bestmodel_20221102.pdf'
        logreg_prc
        plt.savefig(pn)
        plt.close()

        pn = dat_dir + 'bestmodel_plots/metrics_plot_' + model_type + '_' + data_set +'_bestmodel_20221102.pdf'
        visualization.plot_performance_metrics(lr_results, model_type)
        plt.savefig(pn)
        plt.close()

    if model_type == 'random_forest':
        rf_results = {}

        dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'
        dat_file = 'gridsearch_results/' + model_type + '_' + data_set + '_grid_results_20221102.pkl.gz'
        fn = dat_dir + dat_file

        with gzip.open(fn, 'rb') as f:
            rf_search_results = pkl.load(f)

        ranfor_bestparams = rf_search_results['best_params']

        if utils.rf_bestmodel is None:
            # Random forest
            # Train models using best hyperparameters
            ranfor_model = evaluate.get_model(data_dict['Xtrain'],
                                              data_dict['ytrain'],
                                              'random_forest',
                                              ranfor_bestparams,
                                              seed=8)

            rf_results['model'] = ranfor_model

            dat_file = 'bestmodel_results/' + model_type + '_' + data_set + '_best_model_performance_20221102.pkl.gz'
            fn = dat_dir + dat_file

            with gzip.open(fn, 'wb') as f:
                pkl.dump(rf_results, f)

        else:
            with gzip.open(fn, 'rb') as f:
                rf_results = pkl.load(f)

        for seed in range(100):
            Xval, yval = resample(data_dict['Xval'],
                                  data_dict['yval'],
                                  random_state=seed,
                                  stratify=data_dict['yval'])

            # Random Forest Performance Metrics
            ranfor_pred = rf_results['model'].predict(Xval)
            ranfor_metrics = evaluate.get_metrics(yval, ranfor_pred)
            rf_results[seed] = ranfor_metrics

        # roc and prc plots for validation data
        ranfor_roc, ranfor_prc = visualization.plot_curves(rf_results['model'], data_dict['Xval'], data_dict['yval'])

        pn = dat_dir + 'bestmodel_plots/ROC_plot_' + model_type + '_' + data_set +'_bestmodel_20221102.pdf'
        ranfor_roc
        plt.savefig(pn)
        plt.close()

        pn = dat_dir + 'bestmodel_plots/PRC_plot_' + model_type + '_' + data_set +'_bestmodel_20221102.pdf'
        ranfor_prc
        plt.savefig(pn)
        plt.close()

        pn = dat_dir + 'bestmodel_plots/metrics_plot_' + model_type + '_' + data_set +'_bestmodel_20221102.pdf'
        visualization.plot_performance_metrics(rf_results, model_type)
        plt.savefig(pn)
        plt.close()
