import helper
from datetime import date
# import mysql.connector as conn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, required=True)
    args = parser.parse_args()
    g = args.group

    # filenames
    workdir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
    ecg_fn = workdir + 'ecg_mrn_filenames.csv'
    ehr_cihd_fn = workdir + 'msdw_results_cIHD_20220209.tsv'
    ehr_control_putative_fn = workdir + 'putative_controls_ehr_filtered_20220309.pkl.gz'
    case_fn = workdir + 'cases_' + g + '_measurement-demographics_' + str(date.today()) + '.pkl.gz'
    control_fn = workdir + 'controls_' + g + '_measurement-demographics_' + str(date.today()) + '.pkl.gz'
    tplus_fn = workdir + 'cohort_' + g + '_input_dict_' + str(date.today()) + '.pkl.gz'
    tminus_fn = workdir + 'cohort_' + g + '_input_dict_' + str(date.today()) + '.pkl.gz'

    ## to do Step 0: get case and control EHR from HPIMS msdw
    # mydb = conn.connect(
    #     host="localhost",
    #     user="yourusername",
    #     password="yourpassword"
    # )

    # Step 1: get cases
    ecg_df = helper.clean_ecg(utils.ecg_fn) # clean ecg demographics file
    msdw_df = helper.clean_msdw(utils.ehr_cihd_fn) # clean ehr demographics file
    msdw_df = msdw_df[(msdw_df.CALENDAR_DATE < '2020-04-01') | (msdw_df.CALENDAR_DATE > '1950-01-01')] # restrict time
    ## to do: save file of putative control ****
    race_ethnicity_map = helper.map_groups(msdw_df, 'case') # standardize race ethnicity groups
    msdw_df = helper.get_cad_df(msdw_df) # select CAD patients from dataset (move higher up?)
    case_df = pd.merge(ecg_df, msdw_df, on='MEDICAL_RECORD_NUMBER', how='inner') # merge ecg and ehr dataframes
    case_df.loc[:, 'label'] = '1'
    case_df.reset_index(drop=True, inplace=True)
    case_df = helper.select_timeframes(case_df, g) # select timeframe from ecg to dx (move higher up?)
    case_df = helper.get_ecg_measurements(case_df) # extract ecg measurement data from selected cohort
    case_dataset = helper.drop_vars_obs(case_df) # remove rows missing too many variables, variables without enough values
    with gzip.open(case_fn, 'wb') as f: # save case dataset
        pkl.dump(case_dataset, f)

    # Step 2: get controls
    ## to do: get control EHR from HPIMS msdw
    msdw_df = helper.clean_msdw(utils.ehr_control_putative_fn) # clean ehr demographics file
    msdw_df = helper.map_groups(msdw_df, 'control') # standardize race ethnicity groups
    control_df = pd.merge(ecg_df, msdw_df, on='MEDICAL_RECORD_NUMBER', how='inner') # merge ecg and ehr dataframes
    control_df.loc[:, 'label'] = '0'
    control_df = control_df.reset_index(drop=True)
    control_df = helper.matched_controls(case_dataset, control_df) # select controls to match cases
    control_df = helper.get_ecg_measurements(control_df) # extract ecg measurement data from selected cohort
    control_dataset = helper.drop_vars_obs(control_df) # remove rows missing too many variables, variables without enough values
    with gzip.open(control_fn, 'wb') as f: # save control dataset
        pkl.dump(control_dataset, f)

    # Step 3: get t-plus dataset
    if g == 't-plus1':
        tplus_data, tplus_demog = helper.separate_data_demog(case_dataset, control_dataset)
        tplus_data = helper.quantile_filter(tplus_data, 0.0005, 0.9995) # remove outliers
        tplus_demog = tplus_demog.loc[tplus_data.index] # drop demographic rows to match
        tplus_df = pd.concat([tplus_data, tplus_demog], axis=1) # rejoin data and demographics
        tplus_df = helper.even_out(tplus_df) # even out cases and controls
        tplus_data_dict, group_split_dict, tplus_demog_dict = helper.split_sets_tplus(tplus_df) # split dataset
        tplus_data_dict = helper.impute_knn(tplus_data_dict, utils.imputation_type) # impute missing values
        tplus_data_dict = helper.scale_minmax(tplus_data_dict, utils.scaling_type) # normalize input
        tplus_dict = {'data_dict' : tplus_data_dict, 'demog_dict' : tplus_demog_dict} # save tplus dataset
        with gzip.open(tplus_fn, 'wb') as f:
            pkl.dump(tplus_dict, f)

    # Step 4: get t-minus dataset
    else: # load previously saved tplus dataset
        with gzip.open(tplus_fn, 'rb') as f:
            tplus_dict = pkl.load(f)
        tplus_test_df = tplus_dict['demog_dict']['test'] # use test from tplus set to exclude from training tminus
        tminus_data, tminus_demog = helper01.separate_data_demog(case_dataset, control_dataset)
        tminus_data = helpers01.quantile_filter(tminus_data, 0.0005, 0.9995) # remove outliers
        tminus_demog = tminus_demog.loc[tminus_data.index] # drop demographic rows to match
        tminus_df = pd.concat([tminus_data, tminus_demog], axis=1) # rejoin data and demographics
        tminus_data_dict, tminus_demog_dict = split_sets_tminus(tminus_df, tplus_test_df) # split dataset
        tminus_data_dict = helpers01.impute_knn(tminus_data_dict, utils.imputation_type) # impute missing values
        tminus_data_dict = helpers01.scale_minmax(tminus_data_dict, utils.scaling_type) # normalize input
        tminus_dict = {'data_dict': tminus_data_dict, 'demog_dict': tminus_demog_dict} # save tminus dataset
        with gzip.open(tminus_fn, 'wb') as f:
            pkl.dump(tminus_dict, f)
