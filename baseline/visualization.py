
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--version', type=str, required=True)
args = parser.parse_args()
data_set = args.dataset
model_type = args.model
version_date = args.version

# hyperparam search
result_fn = 'gridsearch_results/' + model_type + '_' + data_set + '_grid_results_' + version_date + '.pkl.gz'
f1plot_fn = 'gridsearch_plots/' + model_type + '_' + data_set + '_f1score_gridsearch_plot_' + version_date + '.pdf'
lossplot_fn = 'gridsearch_plots/' + model_type + '_' + data_set + '_loss_gridsearch_plot_' + version_date + '.pdf'
# training eval
train_fn = 'bestmodel_results/' + model_type + '_' + data_set + '_model_performance_' + version_date + '.pkl.gz'
roc_fn = dat_dir + 'bestmodel_plots/ROC_plot_' + model_type + '_' + data_set + '_bestmodel_' + version_date + '.pdf'
prc_fn = dat_dir + 'bestmodel_plots/PRC_plot_' + model_type + '_' + data_set + '_bestmodel_' + version_date + '.pdf'
metric_fn = dat_dir + 'bestmodel_plots/metrics_plot_' + model_type + '_' + data_set + '_bestmodel_' + version_date + '.pdf'

if pipeline_stage == 'hyperparameter_search':
    ## logistic regression
    if model_type == 'logistic_regression':
        with gzip.open(result_fn, 'rb') as f:
            lr_search_results = pkl.load(f)
        lr_performance = helper.get_hyperparam_performance(lr_search_results, model_type)
        helper.plot_grid_search_performance(lr_performance['f1'], search_type, model_type, 'f1')
        plt.savefig(f1plot_fn)
        plt.close()
        helper.plot_grid_search_performance(lr_performance['loss'], search_type, model_type, 'loss')
        plt.savefig(lossplot_fn)
        plt.close()
    if model_type == 'random_forest':
        with gzip.open(result_fn, 'rb') as f:
            rf_search_results = pkl.load(f)
        rf_performance = helper.get_hyperparam_performance(rf_search_results, model_type)
        helper.plot_grid_search_performance(rf_performance['f1'], search_type, model_type, 'f1')
        plt.savefig(f1plot_fn)
        plt.close()

elif pipeline_stage == 'training_evaluation':
    if model_type == 'logistic_regression':
        with gzip.open(train_fn, 'rb') as f:
            lr_results = pkl.load(f)
        # lr viz
        # roc and prc plots for validation data
        logreg_roc, logreg_prc = visualization.plot_curves(lr_results['model'], data_dict['Xval'], data_dict['yval'])
        logreg_roc
        plt.savefig(roc_fn)
        plt.close()
        logreg_prc
        plt.savefig(prc_fn)
        plt.close()
        visualization.plot_performance_metrics(lr_results, model_type)
        plt.savefig(metric_fn)
        plt.close()
    if model_type == 'random_forest':
        with gzip.open(train_fn, 'rb') as f:
            rf_results = pkl.load(f)
        ranfor_roc, ranfor_prc = visualization.plot_curves(rf_results['model'], data_dict['Xval'], data_dict['yval'])
        ranfor_roc
        plt.savefig(pn)
        plt.close()
        ranfor_prc
        plt.savefig(pn)
        plt.close()
        visualization.plot_performance_metrics(rf_results, model_type)
        plt.savefig(pn)
        plt.close()


