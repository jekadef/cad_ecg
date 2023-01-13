# filenames
workdir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
datadir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/input_data/'
ecg_fn = workdir + 'ecg_mrn_filenames.csv'
ehr_cihd_fn = workdir + 'msdw_results_cIHD_20220209.tsv'
ehr_control_putative_fn = workdir + 'putative_controls_ehr_filtered_20220309.pkl.gz'
ethmap_fn = workdir + 'ethnicity_mapping_2022.csv'
racemap_fn = workdir + 'race_mapping_2022.csv'
case_fn = datadir + 'cases_' + g + '_measurement-demographics_' + str(date.today()) + '.pkl.gz'
control_fn = datadir + 'controls_' + g + '_measurement-demographics_' + str(date.today()) + '.pkl.gz'
tplus_fn = datadir + 'cohort_' + g + '_input_dict_' + str(date.today()) + '.pkl.gz'
tminus_fn = datadir + 'cohort_' + g + '_input_dict_' + str(date.today()) + '.pkl.gz'


# files
data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'
input_fn = 'input_data/cohort_' + data_set + '_input_dict_' + version + '.pkl.gz'
search_fn = 'gridsearch_results/' + model_type + '_' + data_set + '_grid_results_' + version + '.pkl.gz'
train_fn = 'bestmodel_results/' + model_type + '_' + data_set + '_model_performance_' + version_date + '.pkl.gz'

lr_grid = {'C': [100, 10, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
           'penalty': ['l2', 'l1']}
rf_grid = {'n_estimators': [int(x) for x in np.linspace(start=10, stop=300, num=4)],
           'max_depth': [int(x) for x in np.linspace(1, 80, num=4)],
           'max_features': ['auto', 'sqrt'],
           'min_samples_leaf': [1, 2, 4, 8],
           'min_samples_split': [2, 5, 10]}

imputation_type = 'median'
scaling_type = 'minmax'

search_type = 'random' # 'grid'

# files
data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'

test_eval_fn = data_dir + 'cohort_' + test_set + '_input_dict_' + version_date + '.pkl.gz'
train_eval_fn = dat_dir + 'bestmodel_results/' + model_type + '_' + train_set + '_best_model_performance_' + version_date + '.pkl.gz'
eval_fn = 'bestmodel_results/' + model_type + '_performance_' + train_set + '_evaluated_on_' + test_set + ' ' + version_date + '.pkl.gz'
