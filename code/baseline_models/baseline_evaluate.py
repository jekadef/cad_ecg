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
    # parser.add_argument('--trainset', type=str, required=True)
    # parser.add_argument('--testset', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    model_type = args.model
    # train_set = args.trainset
    # test_set = args.testset

    train_set = ['t_plus_1', 't_minus_1', 't_minus_5', 't_minus_10', 't_plus_1', 't_plus_1', 't_plus_1']
    test_set = ['t_plus_1', 't_minus_1', 't_minus_5', 't_minus_10', 't_minus_1', 't_minus_5', 't_minus_10']

    results_dict = {}
    for i in range(7):
        dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'
        fn = dat_dir + 'bestmodel_results/' + model_type + '_' + train_set[i] + '_best_model_performance_20221102.pkl.gz'
        with gzip.open(fn, 'rb') as f:
            trained_results = pkl.load(f)
        # feat_importance = trained_results['model'].feature_importances_
        data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
        fn = data_dir + 'cohort_' + test_set[i] + '_input_dict_20221102.pkl.gz'
        with gzip.open(fn, 'rb') as f:
            test_input_dict = pkl.load(f)
        test_data_dict = test_input_dict['data_dict']
        test_demog_dict = test_input_dict['demog_dict']
        # plot_suffix = model_type + '_trained-on_' + train_set + 'evaluated-on_' + test_set + '_bestmodel_20220912.pdf'
    # if model_type == 'logistic_regression':
        mod_results = pd.DataFrame()
        test_results = pd.DataFrame()
        for seed in range(10):
            Xtest, ytest = resample(test_data_dict['Xtest'],
                                  test_data_dict['ytest'],
                                  random_state=seed,
                                  stratify=test_data_dict['ytest'])
            # Logistic Regression Performance Metrics
            mod_pred = trained_results['model'].predict(Xtest)
            mod_metrics = evaluate.get_metrics(ytest, mod_pred)
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
            mod_metrics = evaluate.get_metrics(yval, mod_pred)
            # lr_results = pd.concat([lr_results, pd.Series(logreg_metrics['f1-score'])])
            mod_results = pd.concat([mod_results, pd.DataFrame.from_dict(mod_metrics, orient='index')], axis=1)
        val_results['val'] = round(pd.DataFrame(mod_results.mean(axis=1)), 3)
        results_dict[str(train_set[i]) + ' eval on ' + str(test_set[i])] = pd.concat([test_results, val_results], axis=1)

    print(results_dict)


    # if model_type == 'random_forest':
    #     rf_results = pd.DataFrame()
    #     for seed in range(10):
    #         Xval, yval = resample(test_data_dict['Xval'],
    #                               test_data_dict['yval'],
    #                               random_state=seed,
    #                               stratify=test_data_dict['yval'])
    #
    #         # Random Forest Performance Metrics
    #         ranfor_pred = trained_results['model'].predict(Xval)
    #         ranfor_metrics = evaluate.get_metrics(yval, ranfor_pred)
    #         # rf_results = pd.concat([rf_results, pd.Series(ranfor_metrics['f1-score'])])
    #         rf_results = pd.concat([rf_results, pd.DataFrame.from_dict(ranfor_metrics, orient='index')], axis=1)
    #         print(seed)
    #     rf_results['mean_rows'] = rf_results.mean(axis=1)
    #     print('trained on ' + str(train_set) + ' eval on ' + str(test_set))
    #     print('validation  set')
    #     for i in range(6):
    #         print(rf_results.index[i] + ': ' + str(round(rf_results.loc[rf_results.index[i], 'mean_rows'], 3)))
    #
    #     rf_results = pd.DataFrame()
    #     for seed in range(10):
    #         Xtest, ytest = resample(test_data_dict['Xtest'],
    #                               test_data_dict['ytest'],
    #                               random_state=seed,
    #                               stratify=test_data_dict['ytest'])
    #         # Random Forest Performance Metrics
    #         ranfor_pred = trained_results['model'].predict(Xtest)
    #         ranfor_metrics = evaluate.get_metrics(ytest, ranfor_pred)
    #         # rf_results = pd.concat([rf_results, pd.Series(ranfor_metrics['f1-score'])])
    #         rf_results = pd.concat([rf_results, pd.DataFrame.from_dict(ranfor_metrics, orient='index')], axis=1)
    #         print(seed)
    #     rf_results['mean_rows'] = rf_results.mean(axis=1)
    #     print('trained on ' + str(train_set) + ' eval on ' + str(test_set))
    #     print('test set')
    #     for i in range(6):
    #         print(rf_results.index[i] + ': ' + str(round(rf_results.loc[rf_results.index[i], 'mean_rows'], 3)))
    #


