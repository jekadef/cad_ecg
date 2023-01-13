import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import gzip
import sys
import helper

with gzip.open(utils.input_fn, 'rb') as f:
    input_dict = pkl.load(f)
data_dict = input_dict['data_dict']
demog_dict = input_dict['demog_dict']

def _hyperparameter_search(input):
    # uh oh predefined split in tminus... does that exist?
    output = helper.hyperparam_search(input, model_type, utils.search_type)
    with gzip.open(utils.search_fn, 'wb') as f:
        pkl.dump(output, f)

def _train_models(input):
    with gzip.open(utils.search_fn, 'rb') as f:
        search_results = pkl.load(f)
    trained_results = helper.get_model(input, search_results)
    for seed in range(100):
        Xval, yval = resample(input['Xval'],
                              input['yval'],
                              random_state=8,
                              stratify=input['yval'])
        # Logistic Regression Performance Metrics
        pred_results = trained_results['model'].predict(Xval)
        trained_results[seed] = helper.get_metrics(yval, pred_results)
    with gzip.open(utils.train_fn, 'wb') as f:
        pkl.dump(trained_results, f)

def _evaluate_performance():
    with gzip.open(train_eval_fn, 'rb') as f:
        trained_results = pkl.load(f)
    with gzip.open(test_eval_fn, 'rb') as f:
        test_input_dict = pkl.load(f)

    test_data_dict = test_input_dict['data_dict']
    test_demog_dict = test_input_dict['demog_dict']
    mod_results = pd.DataFrame()
    eval_results = pd.DataFrame()

    for seed in range(100):
        Xtest, ytest = resample(test_data_dict['Xtest'],
                                test_data_dict['ytest'],
                                random_state=seed,
                                stratify=test_data_dict['ytest'])
        # Logistic Regression Performance Metrics
        mod_pred = trained_results['model'].predict(Xtest)
        mod_metrics = get_metrics(ytest, mod_pred)
        df = pd.DataFrame(data=mod_metrics, index=[seed])
        mod_results = pd.concat([mod_results, df], axis=0)

    eval_results['test_mean'] = round(mod_results.mean(axis=0), 3)
    eval_results['test_std'] = round(mod_results.std(axis=0), 3)
    mod_results = pd.DataFrame()

    for seed in range(100):
        Xval, yval = resample(test_data_dict['Xval'],
                              test_data_dict['yval'],
                              random_state=seed,
                              stratify=test_data_dict['yval'])
        # Logistic Regression Performance Metrics
        mod_pred = trained_results['model'].predict(Xval)
        mod_metrics = get_metrics(ytest, mod_pred)
        df = pd.DataFrame(data=mod_metrics, index=[seed])
        mod_results = pd.concat([mod_results, df], axis=0)

    eval_results['val_mean'] = round(mod_results.mean(axis=0), 3)
    eval_results['val_std'] = round(mod_results.std(axis=0), 3)
    with gzip.open(utils.eval_fn, 'wb') as f:
        pkl.dump(eval_results, f)

if __name__ == '__main__':
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

    _hyperparameter_search(data_dict)
    _train_models(data_dict)
    _evaluate_performance()