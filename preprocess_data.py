import preprocess
from datetime import date
import sys
import argparse
import pandas as pd
import pickle as pkl
import gzip
import numpy as np
# import mysql.connector as conn

def _get_cases():
    pass
    # Step 1: get cases
    print('step1')
    ecg_df = preprocess.clean_ecg(utils.ecg_fn) # clean ecg demographics file
    msdw_df = preprocess.clean_msdw(utils.ehr_cihd_fn) # clean ehr demographics file
    ## to do: save file of putative control ****
    # should be used to exclude MRNs BEFORE cleaning ehr_cihd_fn
    msdw_df = preprocess.get_cad_df(msdw_df)  # select CAD patients from dataset
    race_ethnicity_map = preprocess.map_groups(msdw_df, 'case', utils.ethmap_fn, utils.racemap_fn) # standardize race ethnicity groups
    case_df = pd.merge(ecg_df, msdw_df, on='MEDICAL_RECORD_NUMBER', how='inner') # merge ecg and ehr dataframes
    # add additional variables
    case_df['time_delta'] = case_df.acquisitiondate - case_df.CALENDAR_DATE
    case_df['years_icd_ecg'] = case_df.time_delta.dt.days / 365
    case_df = list(range(20, 101, 5))
    case_df['age_binned'] = pd.cut(case_df['age'], age_bins)
    case_df.loc[:, 'label'] = '1'
    case_df.reset_index(drop=True, inplace=True)
    case_df = preprocess.select_timeframes(case_df, g) # select timeframe from ecg to dx (move higher up?)
    # to do: randomly select 1 ECG per patient MRN
    case_df = preprocess.get_ecg_measurements(case_df) # extract ecg measurement data from selected cohort
    case_dataset = preprocess.drop_vars_obs(case_df) # remove rows missing too many variables, variables without enough values
    with gzip.open(utils.case_fn, 'wb') as f: # save case dataset
        pkl.dump(case_dataset, f)

def _get_controls():
    pass
    # Step 2: get controls
    print('step2')
    ## to do: get control EHR from HPIMS msdw
    msdw_df = preprocess.clean_msdw(utils.ehr_control_putative_fn) # clean ehr demographics file
    msdw_df = preprocess.map_groups(msdw_df, 'control') # standardize race ethnicity groups
    control_df = pd.merge(ecg_df, msdw_df, on='MEDICAL_RECORD_NUMBER', how='inner') # merge ecg and ehr dataframes
    control_df.loc[:, 'label'] = '0'
    control_df = control_df.reset_index(drop=True)
    # to do: randomly select 1 ECG per patient MRN
    control_df = preprocess.matched_controls(case_dataset, control_df) # select controls to match cases
    control_df = preprocess.get_ecg_measurements(control_df) # extract ecg measurement data from selected cohort
    control_dataset = preprocess.drop_vars_obs(control_df) # remove rows missing too many variables, variables without enough values
    with gzip.open(utils.control_fn, 'wb') as f: # save control dataset
        pkl.dump(control_dataset, f)

def _get_input():
    pass
    # Step 3: get t-plus dataset
    print('step3')
    if g == 't-plus1':
        tplus_data, tplus_demog = preprocess.separate_data_demog(case_dataset, control_dataset)
        tplus_data = preprocess.quantile_filter(tplus_data, 0.0005, 0.9995) # remove outliers
        tplus_demog = tplus_demog.loc[tplus_data.index] # drop demographic rows to match
        tplus_df = pd.concat([tplus_data, tplus_demog], axis=1) # rejoin data and demographics
        tplus_df = preprocess.even_out(tplus_df) # even out cases and controls
        tplus_data_dict, group_split_dict, tplus_demog_dict = preprocess.split_sets_tplus(tplus_df) # split dataset
        tplus_data_dict = preprocess.impute_knn(tplus_data_dict, utils.imputation_type) # impute missing values
        tplus_data_dict = preprocess.scale_minmax(tplus_data_dict, utils.scaling_type) # normalize input
        tplus_dict = {'data_dict' : tplus_data_dict, 'demog_dict' : tplus_demog_dict} # save tplus dataset
        with gzip.open(utils.tplus_fn, 'wb') as f:
            pkl.dump(tplus_dict, f)
    # Step 4: get t-minus dataset
    else:
    # load previously saved tplus dataset
        print('step4')
        with gzip.open(utils.tplus_fn, 'rb') as f:
            tplus_dict = pkl.load(f)
        tplus_test_df = tplus_dict['demog_dict']['test'] # use test from tplus set to exclude from training tminus
        tminus_data, tminus_demog = preprocess.separate_data_demog(case_dataset, control_dataset)
        tminus_data = preprocess.quantile_filter(tminus_data, 0.0005, 0.9995) # remove outliers
        tminus_demog = tminus_demog.loc[tminus_data.index] # drop demographic rows to match
        tminus_df = pd.concat([tminus_data, tminus_demog], axis=1) # rejoin data and demographics
        tminus_data_dict, tminus_demog_dict = preprocess.split_sets_tminus(tminus_df, tplus_test_df) # split dataset
        tminus_data_dict = preprocess.impute_knn(tminus_data_dict, utils.imputation_type) # impute missing values
        tminus_data_dict = preprocess.scale_minmax(tminus_data_dict, utils.scaling_type) # normalize input
        tminus_dict = {'data_dict': tminus_data_dict, 'demog_dict': tminus_demog_dict} # save tminus dataset
        with gzip.open(utils.tminus_fn, 'wb') as f:
            pkl.dump(tminus_dict, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, required=True)
    parser.add_argument('--filter', type=bool, action='store_false') # ***
    args = parser.parse_args()
    g = args.group

    _get_cases()
    _get_controls()
    _get_input()

    ## to do Step 0: get case and control EHR from HPIMS msdw
    # mydb = conn.connect(
    #     host="localhost",
    #     user="yourusername",
    #     password="yourpassword"
    # )








