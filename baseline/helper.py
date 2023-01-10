import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split, PredefinedSplit, GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import gzip
import pickle as pkl
import pandas as pd

def get_sets(data_df):
    """Takes the dataset and splits into training, validation, test and predefinted splits"""
    data_sets = {}
    group_split = {}
    demographics = {}
    gss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=8)

    x_data = data_df.loc[:, ['VentricularRate', 'AtrialRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
                             'PAxis', 'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset']]
    y_data = data_df.loc[:, 'label']
    demog = data_df.loc[:, ['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate', 'MEDICAL_RECORD_NUMBER',
                            'PERSON_KEY', 'GENDER', 'RACE', 'PATIENT_ETHNIC_GROUP', 'DATE_OF_BIRTH', 'CALENDAR_DATE',
                            'GROUP_RACE_ETHNICITY', 'META_GROUP', 'full_path', 'label', 'time_delta', 'years_icd_ecg',
                            'age_binned', 'weights', 'CONTEXT_DIAGNOSIS_CODE']]

    for temp_ix, test_ix in gss1.split(x_data, y_data):

        data_sets['Xtest'] = x_data.iloc[test_ix].to_numpy()
        data_sets['ytest'] = y_data.iloc[test_ix].to_numpy()
        # group_split['test'] = group.iloc[test_ix]
        demographics['test'] = demog.iloc[test_ix]

        x_temp = x_data.iloc[temp_ix]
        y_temp = y_data.iloc[temp_ix]
        # group_temp = group.iloc[temp_ix]
        demog_temp = demog.iloc[temp_ix]

        data_sets['Xtemp'] = x_data.iloc[temp_ix].to_numpy()
        data_sets['ytemp'] = y_data.iloc[temp_ix].to_numpy()
        # group_split['temp'] = group.iloc[temp_ix]
        demographics['temp'] = demog.iloc[temp_ix]

    gss2 = StratifiedShuffleSplit(n_splits=1, train_size=.75, random_state=8)
    for train_ix, val_ix in gss2.split(x_temp, y_temp):

        split_index = [-1 if x in train_ix else 0 for x in temp_ix]
        ps = PredefinedSplit(test_fold=split_index)
        data_sets['predef_split'] = ps

        data_sets['Xtrain'] = x_temp.iloc[train_ix].to_numpy()
        data_sets['ytrain'] = y_temp.iloc[train_ix].to_numpy()
        # group_split['train'] = group_temp.iloc[train_ix]
        demographics['train'] = demog_temp.iloc[train_ix]

        data_sets['Xval'] = x_temp.iloc[val_ix].to_numpy()
        data_sets['yval'] = y_temp.iloc[val_ix].to_numpy()
        # group_split['val'] = group_temp.iloc[val_ix]
        demographics['val'] = demog_temp.iloc[val_ix]

    return data_sets, group_split, demographics

def lr_search(search_type, df, grid):
    """ Identifying best parameters"""
    x_data = df['Xtemp']
    y_data = df['ytemp']
    data_splits = df['predef_split']
    results_dict = {} # lr = LogisticRegression(max_iter=500, solver='saga', random_state=8)
    lr = LogisticRegression(max_iter=500, solver='saga', random_state=8)
    scoring = {'f1': 'f1', 'loss': 'neg_log_loss'}
    if search_type == 'random':
        logreg_grid = RandomizedSearchCV(estimator=lr,
                                         param_distributions=grid,
                                         cv=data_splits,
                                         scoring=scoring,
                                         verbose=3,
                                         refit='f1',
                                         error_score=0,
                                         return_train_score=True,
                                         random_state=8)

    if search_type == 'grid':
        logreg_grid = GridSearchCV(estimator=lr,
                                   param_grid=grid,
                                   cv=data_splits,
                                   scoring=scoring,
                                   verbose=3,
                                   refit='f1',
                                   error_score=0,
                                   return_train_score=True)
    logreg_grid.fit(x_data, y_data)
    results_dict = {#'best_estimator': logreg_grid.best_estimator_,
                    'best_params': logreg_grid.best_params_,
                    'performance_results': logreg_grid.cv_results_}
    return results_dict

def rf_search(search_type, df, grid):
    """ Identifying best parameters"""
    x_data = df['Xtemp']
    y_data = df['ytemp']
    data_splits = df['predef_split']
    results_dict = {}
    rf = RandomForestClassifier()

    if search_type == 'random':
        rf_grid = RandomizedSearchCV(estimator=rf,
                                     param_distributions=grid,
                                     n_iter=30,
                                     cv=data_splits,
                                     scoring='f1',
                                     verbose=3,
                                     refit=False,
                                     error_score=0,
                                     return_train_score=True,
                                     random_state=8)

    if search_type == 'grid':
        rf_grid = GridSearchCV(estimator=rf,
                               param_grid=grid,
                               cv=data_splits,
                               scoring='f1',
                               verbose=3,
                               refit=False,
                               error_score=0,
                               return_train_score=True)
    rf_grid.fit(x_data, y_data)
    results_dict = {#'best_estimator': rf_grid.best_estimator_,
                    'best_params': rf_grid.best_params_,
                    'performance_results': rf_grid.cv_results_}

    return results_dict

def load_process_data(df, drop_vars):
    demographics = df[['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate', 'MEDICAL_RECORD_NUMBER',
                    'PERSON_KEY', 'GENDER', 'RACE', 'PATIENT_ETHNIC_GROUP', 'DATE_OF_BIRTH', 'CALENDAR_DATE',
                    'GROUP_RACE_ETHNICITY', 'META_GROUP', 'full_path', 'time_delta', 'years_icd_ecg',
                    'age_binned', 'weights', 'CONTEXT_DIAGNOSIS_CODE']]
    # variables to include in the model
    data = df[['label', 'VentricularRate', 'AtrialRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
               'PAxis', 'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset',
               'ECGSampleBase', 'ECGSampleExponent', 'QTcFrederica', 'SystolicBP', 'DiastolicBP']]
    data = data.astype(float)
    data['label'] = data['label'].astype(int)
    # drop variables that were chosen in the missingness analysis
    data = data.drop(drop_vars, axis='columns')
    return demographics, data

def quantile_filter(x_data, low, high):
    low_quantile = x_data.quantile(low, axis='rows')
    high_quantile = x_data.quantile(high, axis='rows')

    for col in x_data.columns:
        x_data = x_data.loc[((x_data[col] >= low_quantile[col]) & (x_data[col] <= high_quantile[col])) | x_data[col].isna()]

    return x_data

def get_imputed(x_data, impute_type):
    """ Takes the dataset with missing data and returns an imputed dataset"""
    # imputer = SimpleImputer(strategy=impute_type)
    imputer = KNNImputer()
    imputed_data = imputer.fit_transform(x_data)
    return imputer, imputed_data

def get_scaling(x_data, scale_type):
    """Scale data"""
    if scale_type == 'standard':
        scaler = StandardScaler()
    elif scale_type == 'minmax':
        scaler = MinMaxScaler()
    scaler.fit(x_data)
    scaled_data = scaler.transform(x_data)
    return scaler, scaled_data

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

def get_lr_model(df, search_results):
    """ Trains a model and makes predictions """
    results = {}
    best_params = search_results['best_params']
    train_data = df['Xtrain']
    train_label = df['ytrain']
    model = LogisticRegression(C=best_params['C'],
                               penalty=best_params['penalty'],
                               max_iter=500,
                               solver='saga',
                               random_state=8)
    model.fit(train_data, train_label)
    results['model'] = model
    results['coefficient'] = model.coef_
    results['beta'] = model.intercept_
    return results

def get_rf_model(df, search_results):
    """ Trains a model and makes predictions """
    results = {}
    best_params = search_results['best_params']
    train_data = df['Xtrain']
    train_label = df['ytrain']
    model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                   max_depth=best_params['max_depth'],
                                   min_samples_leaf=best_params['min_samples_leaf'],
                                   min_samples_split=best_params['min_samples_split'],
                                   max_features=best_params['max_features'],
                                   bootstrap=True,
                                   random_state=8)
    model.fit(train_data, train_label)
    results['model'] = model
    results['feat_import'] = model.feature_importances_

    return results

def plot_distributions(filename, data):
    for col in data.columns:
        res = data.dropna(subset=[col])
        plot = sns.displot(res, x=col, hue='label', kde=True, aspect=2.5)
        plt.axvline(res[col].mean(), color='k', linestyle='dashed', linewidth=1.5)
        plt.axvline(res[col].median(), color='r', linestyle='dashed', linewidth=1.5)
        plot.savefig(filename + col + '.png')
    return

def plot_curves(model, x_data, y_data):
    roc = plot_roc_curve(model, x_data, y_data)
    prc = plot_precision_recall_curve(model, x_data, y_data)

    return roc, prc

def plot_multiple_curves(model_type, which_curve):
    time_frames = ['t_after_from0to1', 't_before_from-1to0', 't_before_from-5to-1', 't_before_from-10to-5']

    # set up plotting area
    plt.figure(0).clf()

    for t in time_frames:

        dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'
        fn = dat_dir + 'bestmodel_results/' + model_type + '_' + t + '_best_model_performance_20220912.pkl.gz'
        with gzip.open(fn, 'rb') as f:
            trained_results = pkl.load(f)

        data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/cohort_data/'
        fn = data_dir + 'cohort_' + t + '_input_dict_20220912.pkl.gz'
        with gzip.open(fn, 'rb') as f:
            test_input_dict = pkl.load(f)

        Xdata = test_input_dict['data_dict']['Xval']
        ydata = test_input_dict['data_dict']['yval']

        # fit logistic regression model and plot ROC curve
        y_pred = trained_results['model'].predict_proba(Xdata)[:, 1]
        if which_curve == 'ROC':
            fpr, tpr, _ = metrics.roc_curve(ydata, y_pred)
            auc = round(metrics.roc_auc_score(ydata, y_pred), 4)
            plt.plot(fpr, tpr, label=t + ", AUC=" + str(auc))

        # # fit gradient boosted model and plot ROC curve
        # y_pred = model.predict_proba(Xdata)[:, 1]
        # fpr, tpr, _ = metrics.roc_curve(Xdata, y_pred)
        # auc = round(metrics.roc_auc_score(ydata, y_pred), 4)
        # plt.plot(fpr, tpr, label="Gradient Boosting, AUC=" + str(auc))

    # add legend
    plt.legend()
    dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/bestmodel_plots/'
    pn = dat_dir + 'all_timeframe_eval-on-self_' + model_type + '_' + which_curve + '_plot.pdf'
    #save plot
    plt.savefig(pn)

def plot_grid_search_performance(performance_df, search_type, classifier_type, metric):
    # if classifier_type == 'random_forest':
    #     p = sns.relplot(x='n_estimators', y='f1_score', style='max_features', kind='line',
    #                     col='max_depth', hue='set', palette='plasma', data=performance_df)
    if classifier_type == 'random_forest':
        if search_type == 'grid':
            p = sns.catplot(x='max_depth', y='f1_score', row='min_samples_split', col='max_features', kind='bar',
                            hue='min_samples_leaf', palette='plasma', data=performance_df)
        if search_type == 'random':
            p = sns.catplot(x='max_depth', y='f1_score', row='min_samples_split', col='max_features', kind='bar',
                            hue='min_samples_leaf', palette='plasma', data=performance_df)


    elif classifier_type == 'logistic_regression':
        if search_type == 'grid':
            p = sns.relplot(x='C', y=str(metric + '_score'), hue='penalty', style='set', kind='line',
                            palette='plasma', data=performance_df)
            p.fig.get_axes()[0].set_xscale('log')

        if search_type == 'random':
            p = sns.relplot(x='C', y=str(metric + '_score'), hue='penalty', style='set', kind='line',
                            palette='plasma', data=performance_df)
            p.fig.get_axes()[0].set_xscale('log')
    #p = p.set(ylim=(0.0, 1.0))
    return p

def plot_performance_metrics(results_df, classifier_type):
    bootstrap_results = pd.DataFrame()
    for i in range(100):
        s = pd.Series(results_df[i])
        s = s.transpose()
        s = s.reset_index()
        s.columns = ['metric', 'value']
        s['model'] = i
        bootstrap_results = bootstrap_results.append(s)

    if classifier_type == 'logistic_regression':
        p = sns.catplot(x='metric', y='value', kind='box', palette='plasma', data=bootstrap_results).set(
            title='Logistic Regression Model')

    elif classifier_type == 'random_forest':
        p = sns.catplot(x='metric', y='value', kind='box', palette='plasma', data=bootstrap_results).set(
            title='Random Forest Model')

    p = p.set(ylim=(0.0, 1.0))

    return p

def plot_coefficient(coeff, feature):
    df = pd.DataFrame(coeff)
    df.columns = feature
    df = df.transpose().reset_index()
    df.columns = ['feature', 'beta']

    p = sns.catplot(x='feature', y='beta', kind='bar', palette='plasma', data=df)

    return p

