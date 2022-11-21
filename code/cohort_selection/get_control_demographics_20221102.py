import sys
import argparse
import pandas as pd
import pickle as pkl
import gzip
import numpy as np
sys.path.insert(0, '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/code/cohort_selection')
import mapping
import utils
import process


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, required=True)
    args = parser.parse_args()

    g = args.group
    # group = 't_after'

    # load control ecg data
    with gzip.open(utils.df_control_putative_fn, 'rb') as f:
        controls_putative = pkl.load(f)

    # load ehr data for control patients
    with gzip.open(utils.ehr_control_putative_fn, "rb") as f:
        control_dx = pkl.load(f)
    # get this file by running script get_putative_controls_filtered.py


    # process and filter control ehr data
    control_dx = control_dx[control_dx['DATE_OF_BIRTH'] != '0000-00-00 00:00:00']
    control_dx = control_dx[control_dx['CALENDAR_DATE'] != '0000-00-00 00:00:00']
    control_dx.GENDER = control_dx.GENDER.str.upper()
    control_dx.RACE = control_dx.RACE.str.upper()
    control_dx = control_dx[~(control_dx.GENDER.isna())]
    control_dx = control_dx[control_dx.GENDER.str.contains('MALE')]
    control_dx = control_dx[~control_dx.CALENDAR_DATE.str.contains('21..\-', regex=True)]
    control_dx = control_dx[~control_dx.CALENDAR_DATE.str.contains('25..\-', regex=True)]
    control_dx.DATE_OF_BIRTH = pd.to_datetime(control_dx.DATE_OF_BIRTH)
    control_dx.CALENDAR_DATE = pd.to_datetime(control_dx.CALENDAR_DATE)
    # control_dx = control_dx[(control_dx.CALENDAR_DATE < '2020-04-01') & (control_dx.CALENDAR_DATE > '1950-01-01')]

    # drop duplicate EHRs by MRN
    controls_putative.drop_duplicates(subset=['MEDICAL_RECORD_NUMBER'], inplace=True)
    control_dx.drop_duplicates(subset=['MEDICAL_RECORD_NUMBER'], inplace=True)

    # get race and ethnicity groups for control cohort
    ethnicity_map_control = mapping.get_ethnicity_map(control_dx, 'control')
    race_map_control = mapping.get_race_map(control_dx, 'control')
    race_ethnicity_map_control = mapping.get_race_ethnic_groups(ethnicity_map_control, race_map_control, control_dx)
    control_dx_re = pd.merge(control_dx, race_ethnicity_map_control, on='MEDICAL_RECORD_NUMBER', how='left')

    # get meta-group which is gender + race/ethnicity
    control_dx_re['META_GROUP'] = control_dx_re.GENDER.str.upper() + '_' + control_dx_re.GROUP_RACE_ETHNICITY

    # get control patients
    # # but what does calendar date correspond to? There is no diagnosis
    # --> data entry into ehr
    control_dx_re = control_dx_re.sort_values(by=['CALENDAR_DATE'])
    # control_dx_re = control_dx_re.drop_duplicates(subset=['MEDICAL_RECORD_NUMBER'], keep='first')

    # get control dataset of ehr and ecg data combination
    controls = pd.merge(controls_putative, control_dx_re, on='MEDICAL_RECORD_NUMBER', how='inner')
    controls.loc[:, 'label'] = '0'
    controls = controls.reset_index(drop=True)
    controls = controls[(controls['age'] >= 20) & (controls['age'] <= 90)]
    controls['full_path'] = controls.path + '/' + controls.filename
    controls.full_path = controls.full_path.str.replace('/sharepoint/ecg/', '/sc/arion/projects/mscic1/data/GE_ECG_RSYNC/')
    controls = controls.reset_index(drop=True)

    # time component --> not the exact meaning as with cases since no diagnosis date
    controls['time_delta'] = controls.acquisitiondate - controls.CALENDAR_DATE
    controls['years_icd_ecg'] = controls.time_delta.dt.days / 365

    # create age binned variable
    age_bins = list(range(20, 101, 5))
    controls['age_binned'] = pd.cut(controls['age'], age_bins)

    # create icd-ecg timedelta binned variable
    # three_year_bin = list(range(-39, 46, 3))
    # controls['three_year_binned'] = pd.cut(controls['years_icd_ecg'], three_year_bin)

    # case_set = 't_after_measurement-demographics_20220912'

    # data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/cohort_data/'
    # fn = data_dir + 'cases_split_' + group + '_measurement-demographics_20220912.pkl.gz'
    #
    # with gzip.open(fn, 'rb') as f:
    #     cases_dict = pkl.load(f)
    #
    # if group == 't_after':
    #     # dict_keys = ['from0to1']
    #     dict_keys = ['from0to1', 'from1to10', 'from10to20']
    #
    # if group == 't_before':
    #     dict_keys = ['from-1to0', 'from-5to-1', 'from-10to-5', 'from-15to-10', 'from-20to-15', 'from-25to-20',
    #                  'from-30to-25']
    #
    # # k = 'from10to20'
    # # g = 'FEMALE_AMERICAN-INDIAN-OR-ALASKA-NATIVE'
    data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'

    fn = data_dir + 'cases_split_' + g + '_measurement-demographics_20221102.pkl.gz'

    with gzip.open(fn, 'rb') as f:
        cases_df = pkl.load(f)


    group_meta = cases_df.META_GROUP.drop_duplicates()
    # timedelta_bins = filtered_cases_dt.three_year_binned.drop_duplicates()
    selected_cases = cases_df
    sampled_controls = pd.DataFrame()
    controls_df = pd.DataFrame()

    for gm in group_meta:
        # print(cases_to_add)
        g_cases = selected_cases[selected_cases['META_GROUP'] == gm]
        ctrl_to_sample = int(len(g_cases) * 1.8) #+ cases_to_add
        # g_cases = g_cases.drop_duplicates(subset=['MEDICAL_RECORD_NUMBER', 'age'])
        # get age weights per ancestry group per timedelta
        group_weight = g_cases['age_binned'].value_counts(sort=False)
        group_weight = group_weight.sort_index()
        group_weight = pd.DataFrame(group_weight)
        group_weight.reset_index(inplace=True)
        group_weight.columns = ['age_binned', 'weights']
        group_weight.loc[group_weight.weights == 0, 'weights'] = 1
        g_controls = controls[controls['META_GROUP'] == gm]
        if (ctrl_to_sample == 0) | (int(len(g_controls)) == 0):
            # cases_to_add = ctrl_to_sample
            continue
        x = g_controls.merge(group_weight, how='left', on='age_binned')
        if int(len(g_controls)) - ctrl_to_sample < 0:
            try:
                y = x.sample(n=len(g_controls), weights='weights', random_state=10)
            except ValueError:
                print('not enough cases')
                continue
            # cases_to_add = ctrl_to_sample - int(len(g_controls))
        else:
            y = x.sample(n=ctrl_to_sample, weights='weights', random_state=10)
        cntl_measurements, cntl_demog = process.get_ecg_measurements(y)
        measurement_df = pd.DataFrame.from_records(cntl_measurements)
        cntl_demog.reset_index(drop=True, inplace=True)
        # data = data.astype(float)
        # data['label'] = data['label'].astype(int)

        variables_dropped = ['SystolicBP', 'DiastolicBP', 'QTcFrederica', 'ECGSampleBase', 'ECGSampleExponent']
        col_to_drop = measurement_df.columns[measurement_df.columns.isin(variables_dropped)]
        measurement_df = measurement_df.drop(col_to_drop, axis='columns')

        cohort_nan = measurement_df.isna().sum(axis='columns')
        # cohort_nan.value_counts()
        # remove samples that have more than 5 out of 15 variables missing
        cohort_impute = cohort_nan[cohort_nan <= 5]

        # Drop demographic rows to match
        measurement_df = measurement_df.loc[cohort_impute.index]
        demographics_df = cntl_demog.loc[cohort_impute.index]

        sampled_df = pd.concat([demographics_df, measurement_df], axis=1)
        sampled_controls = pd.concat([sampled_controls, sampled_df], axis=0)
        # sampled_controls[g] = pd.concat([sampled_controls, y], ignore_index=True)
        missing_controls = len(sampled_df) - len(g_cases)
        print(missing_controls)

    sampled_controls['label'] = sampled_controls['label'].astype(int)

    # save control dataframe
    fn = data_dir + 'controls_split_' + g + '_measurement-demographics_20221102.pkl.gz'
    with gzip.open(fn, 'wb') as f:
        pkl.dump(sampled_controls, f)

    #
    # fn = data_dir + 'metadict_controls_missing' + g + '_measurement-demographics_20221102.pkl.gz'
    # with gzip.open(fn, 'wb') as f:
    #     pkl.dump(missing_controls_dict, f)


