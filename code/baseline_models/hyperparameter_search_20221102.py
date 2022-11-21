import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import gzip
import sys
sys.path.insert(0, '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/code/baseline_models')
import utils
import preprocess
import split
import search
import evaluate
import visualization


#
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    # parser.add_argument('--timeframe', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--searchtype', type=str, required=True)
    args = parser.parse_args()

    model_type = args.model
    # time_frame = args.timeframe
    data_set = args.dataset
    search_type = args.searchtype

    data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
    fn = data_dir + 'cohort_' + data_set + '_input_dict_20221102.pkl.gz'
    with gzip.open(fn, 'rb') as f:
        input_dict = pkl.load(f)

    data_dict = input_dict['data_dict']
    demog_dict = input_dict['demog_dict']

    if model_type == 'logistic_regression':
        dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'
        fn = dat_dir + 'gridsearch_results/logistic_regression_' + data_set + '_grid_results_20221102.pkl.gz'
        # fn = dat_dir + dat_file
        if utils.lr_bestparams is None:

            lr_search_results = search.search(search_type,
                                              data_dict['Xtemp'],
                                              data_dict['ytemp'],
                                              data_dict['predef_split'],
                                              model_type,
                                              utils.lr_grid)
            with gzip.open(fn, 'wb') as f:
                pkl.dump(lr_search_results, f)

        else:
            with gzip.open(fn, 'rb') as f:
                lr_search_results = pkl.load(f)

        lr_performance = evaluate.get_hyperparam_performance(lr_search_results, model_type)
        print(lr_performance)
        visualization.plot_grid_search_performance(lr_performance['f1'], search_type, model_type, 'f1')
        pn = dat_dir + 'gridsearch_plots/logistic_regression_' + data_set + '_f1score_gridsearch_plot_20221102.pdf'
        # pn = str(utils.grid_search_plots).replace('W', model_type).replace('X', 'f1-score').replace('Y', timedelta).replace('Z', str(window_size))
        plt.savefig(pn)
        plt.close()

        lr_performance = evaluate.get_hyperparam_performance(lr_search_results, model_type)
        visualization.plot_grid_search_performance(lr_performance['loss'], search_type, model_type,  'loss')
        # pn = str(utils.grid_search_plots).replace('W', model_type).replace('X', 'loss').replace('Y', timedelta).replace('Z', str(window_size))
        pn = dat_dir + 'gridsearch_plots/logistic_regression_' + data_set + '_loss_gridsearch_plot_20221102.pdf'
        plt.savefig(pn)
        plt.close()

    if model_type == 'random_forest':
        dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'
        fn = dat_dir + 'gridsearch_results/random_forest_' + data_set + '_grid_results_20221102.pkl.gz'
        # fn = dat_dir + dat_file
        if utils.rf_bestparams is None:
            # Search hyperparameters
            rf_search_results = search.search(search_type,
                                              data_dict['Xtemp'],
                                              data_dict['ytemp'],
                                              data_dict['predef_split'],
                                              model_type,
                                              utils.rf_grid)
            with gzip.open(fn, 'wb') as f:
                pkl.dump(rf_search_results, f)

        else:
            with gzip.open(fn, 'rb') as f:
                rf_search_results = pkl.load(f)

        rf_performance = evaluate.get_hyperparam_performance(rf_search_results, model_type)
        print(rf_performance)
        visualization.plot_grid_search_performance(rf_performance['f1'], search_type, model_type, 'f1')
        pn = dat_dir + 'gridsearch_plots/random_forest_' + data_set + '_loss_gridsearch_plot_20221102.pdf'
        plt.savefig(pn)
        plt.close()
