from datetime import date

# cohort_selection_workingdir = '/hpc/users/defrej02/projects/cardio_phenotyping/data/cohort_selection/'
cohort_selection_workingdir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'

ethnicity_fn = cohort_selection_workingdir + 'ethnicity_mapping_2022.csv'
race_fn = cohort_selection_workingdir + 'race_mapping_2022.csv'

ecg_fn = cohort_selection_workingdir + 'ecg_mrn_filenames.csv'

ehr_cihd_fn = cohort_selection_workingdir + 'msdw_results_cIHD_20220209.tsv'

case_measurement_fn = cohort_selection_workingdir + 'dict_measurement_case_X_' + str(date.today()) + '.pkl.gz'
#case_filtered_data_fn = cohort_selection_workingdir + 'dict_filtered_case_df_20220309.pkl.gz'

control_putative_fn = cohort_selection_workingdir + 'putative_control_MRN_20220309.csv'
df_control_putative_fn = cohort_selection_workingdir + 'putative_control_df_20220309.pkl.gz'
ehr_control_putative_fn = cohort_selection_workingdir + 'putative_controls_ehr_filtered_20220309.pkl.gz'

#ehr_control_fn = cohort_selection_workingdir + 'msdw_results_control_20220225.tsv'

ctrl_measurement_fn = cohort_selection_workingdir + 'dict_measurement_control_X_' + str(date.today()) + '.pkl.gz'
#ctrl_filtered_data_fn = cohort_selection_workingdir + 'dict_filtered_control_df_20220309.pkl.gz'

cohort_measurement_fn = cohort_selection_workingdir + 'dict_measurement_cohort_X_' + str(date.today()) + '.pkl.gz'
#cohort_filtered_data_fn = cohort_selection_workingdir + "dict_filtered_cohort_df_20220309.pkl.gz"
