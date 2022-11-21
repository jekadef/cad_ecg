from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import gzip
import pickle as pkl
import pandas as pd


def plot_distributions(filename, data):
    for col in data.columns:
        res = data.dropna(subset=[col])
        plot = sns.displot(res, x=col, hue='label', kde=True, aspect=2.5)
        plt.axvline(res[col].mean(), color='k', linestyle='dashed', linewidth=1.5)
        plt.axvline(res[col].median(), color='r', linestyle='dashed', linewidth=1.5)
        plot.savefig(filename + col + '.png')


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




# fig, ax = plt.subplots()
#
# models = [
#     ("RT embedding -> LR", rt_model),
#     ("RF", random_forest),
#     ("RF embedding -> LR", rf_model),
#     ("GBDT", gradient_boosting),
#     ("GBDT embedding -> LR", gbdt_model),
# ]
#
# model_displays = {}
# for name, pipeline in models:
#     model_displays[name] = RocCurveDisplay.from_estimator(
#         pipeline, X_test, y_test, ax=ax, name=name
#     )
# _ = ax.set_title("ROC curve")


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

