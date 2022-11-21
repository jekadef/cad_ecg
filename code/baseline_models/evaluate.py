import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score


def get_hyperparam_performance(performance_df, model):
    """get performance of hyperparameter search """
    if model == 'random_forest':
        performance_results_dict = {}
        val_df = pd.DataFrame(performance_df['performance_results']['params'])
        val_df['set'] = 'validation'
        val_df['f1_score'] = performance_df['performance_results']['mean_test_score']
        val_df = val_df.reset_index()
        val_df.columns = val_df.columns.str.replace('index', 'model')
        # val_df.columns = ['model', 'max_depth', 'max_features', 'set', 'f1_score']
        #val_df.columns = ['model', 'max_depth', 'max_features', 'n_estimators', 'set', 'f1_score']
        train_df = pd.DataFrame(performance_df['performance_results']['params'])
        train_df['set'] = 'training'
        train_df['f1_score'] = performance_df['performance_results']['mean_train_score']
        train_df = train_df.reset_index()
        train_df.columns = train_df.columns.str.replace('index', 'model')
        # train_df.columns = ['model', 'max_depth', 'max_features', 'set', 'f1_score']
        #train_df.columns = ['model', 'max_depth', 'max_features', 'n_estimators', 'set', 'f1_score']
        performance_results = pd.concat([val_df, train_df], axis=0)
        performance_results = performance_results.reset_index(drop=True)
        performance_results['max_depth'] = performance_results['max_depth'].fillna(0)
        performance_results['max_features'] = performance_results['max_features'].fillna('none')
        performance_results = performance_results.astype(
            {'model': int, 'max_depth': str, 'max_features': str, 'set': str, 'f1_score': float})
        performance_results_dict['f1'] = performance_results
    if model == 'logistic_regression':
        scoring = ['f1', 'loss']
        performance_results_dict = {}
        for score in scoring:
            val_df = pd.DataFrame(performance_df['performance_results']['params'])
            val_df['set'] = 'validation'
            val_df[str(score + '_score')] = performance_df['performance_results'][str('mean_test_' + score)]
            val_df = val_df.reset_index()
            # val_df.columns = ['model', 'C', 'penalty', 'set', str(score + '_score')]
            val_df.columns = val_df.columns.str.replace('index', 'model')
            train_df = pd.DataFrame(performance_df['performance_results']['params'])
            train_df['set'] = 'training'
            train_df[str(score + '_score')] = performance_df['performance_results'][str('mean_train_' + score)]
            train_df = train_df.reset_index()
            # train_df.columns = ['model', 'C', 'penalty', 'set', str(score + '_score')]
            train_df.columns = train_df.columns.str.replace('index', 'model')
            performance_results = pd.concat([val_df, train_df], axis=0)
            performance_results = performance_results.reset_index(drop=True)
            performance_results = performance_results.astype(
                {'model': int, 'C': float, 'penalty': str, 'set': str, str(score + '_score'): float})
            performance_results_dict[str(score)] = performance_results
    return performance_results_dict


def get_metrics(true, predicted):
    """Returns accuracy, precision, recall, f1-score, AUC-ROC, AUC-PR(Average Precision)"""
    metrics_dict = {}
    report = classification_report(true, predicted, output_dict=True)
    metrics_dict['accuracy'] = report['accuracy']
    metrics_dict['precision'] = report['1']['precision']
    metrics_dict['recall'] = report['1']['recall']
    metrics_dict['f1-score'] = report['1']['f1-score']
    metrics_dict['AUC-ROC'] = roc_auc_score(true, predicted)
    metrics_dict['AUC-PR'] = average_precision_score(true, predicted)

    return metrics_dict


def get_model(train_data, train_label, model_type, best_params, seed):
    """ Trains a model and makes predictions """
    if model_type == "logistic_regression":
        model = LogisticRegression(C=best_params['C'],
                                   penalty=best_params['penalty'],
                                   max_iter=500,
                                   solver='saga',
                                   random_state=8)
        model.fit(train_data, train_label)

    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                       max_depth=best_params['max_depth'],
                                       min_samples_leaf=best_params['min_samples_leaf'],
                                       min_samples_split=best_params['min_samples_split'],
                                       max_features=best_params['max_features'],
                                       bootstrap=True,
                                       random_state=seed)
        model.fit(train_data, train_label)

    return model

