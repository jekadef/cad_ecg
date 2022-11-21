import numpy as np
from datetime import date

data_folder = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'
cohort_folder = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'

cohort_file_name = cohort_folder + 'dict_measurement_cohort_window_size_Z2022-08-25.pkl.gz'

measurement_file_name = cohort_folder + 'dict_measurement_cohort_timedelta_Y_windowsize_Z_2022-08-26.pkl.gz'
demographics_file_name = cohort_folder + 'dict_demographics_split_cohort_timedelta_Y_windowsize_Z_2022-08-26.pkl.gz'
# path_label_file_name = cohort_folder + 'dict_cohort_filenames_labels_timedelta_Y_windowsize_Z_2022-08-26.pkl.gz'

variables_dropped = ['SystolicBP', 'DiastolicBP', 'QTcFrederica', 'ECGSampleBase', 'ECGSampleExponent']

timedeltas = ['t_after', 't_before', 't_zero']
# windows = ['91', '183', '365']
windows = ['365']

imputation_type = 'median'
scaling_type = 'minmax'


lr_grid = {'C': [100, 10, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
           'penalty': ['l2', 'l1']}

# el_lr_grid = {'l1_ratio': [0, 0.25, 0.5, 0.75, 1]}

# lr_grid = {'C': [100, 10, 1, 0.1, 0.01, 0.001],
#           'penalty': ['none', 'l2', 'l1', 'elasticnet']}

# rf_grid = {#'n_estimators': [10, 50, 100],
#            'max_features': ['sqrt', None],
#            'max_depth': [None, 10, 100]}

# rf_grid = {'n_estimators': [50, 100, 250, 500],
#            'max_depth': [5, 25, 50, 75, 100, None],
#            'max_features': ['auto', 'sqrt'],
#            'min_samples_leaf': [1, 2, 4, 8],
#            'min_samples_split': [2, 5, 10, 20]}

rf_grid = {'n_estimators': [int(x) for x in np.linspace(start=10, stop=300, num=4)],
           'max_depth': [int(x) for x in np.linspace(1, 80, num=4)],
           'max_features': ['auto', 'sqrt'],
           'min_samples_leaf': [1, 2, 4, 8],
           'min_samples_split': [2, 5, 10]}


lr_bestparams = None
rf_bestparams = None
grid_results = data_folder + 'gridsearch_results/X_grid_results_timedelta_Y_windowsize_Z_2022-08-27.pkl.gz'
grid_search_plots = data_folder + 'gridsearch_plots/X_gridsearch_plot_timedelta_Y_windowsize_Z_'+ str(date.today()) + '.pdf'

lr_bestmodel = None
rf_bestmodel = None
performance_results = data_folder + 'bestmodel_results/X_bestmodel_performance_timedelta_Y_windowsize_Z_'+ str(date.today()) +'.pkl.gz'
performance_plots = data_folder + 'bestmodel_plots/W_plot_X_bestmodel_timedelta_Y_windowsize_Z_' + str(date.today()) + '.pdf'
aggregate_plots = data_folder + 'bestmodel_plots/W_aggregate_X_plot_' + str(date.today()) + '.pdf'

# plot_folder = '/hpc/users/defrej02/projects/cardio_phenotyping/data/baseline_results/'
#ranfor_grid_results = data_folder + 'ranfor_grid_results_X_' + str(date.today()) + '.pkl.gz'
#rf_grid_search_plots = plot_folder + 'timedelta_X_ranfor_gridsearch_' + str(date.today()) + '.pdf'
#ranfor_performance = data_folder + 'timedelat_X_ranfor_performance_' + str(date.today()) + '.pkl.gz'
#rf_performance_plots = plot_folder + 'timedelta_X_ranfor_Y_' + str(date.today()) + '.pdf'
