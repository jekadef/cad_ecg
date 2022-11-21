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
    # parser.add_argument('--timestart', type=int, required=True)
    # parser.add_argument('--timestop', type=int, required=True)
    args = parser.parse_args()

    g = args.group

    # load and process and filter ecg data
    ecg_name = pd.read_csv(utils.ecg_fn, header=None)
    ecg_name.columns = ['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate']

    ecg_name['MEDICAL_RECORD_NUMBER'] = ecg_name['patientid'].str.lstrip('0')
    ecg_name = ecg_name[~(ecg_name.MEDICAL_RECORD_NUMBER.isna())]
    ecg_name = ecg_name[~(ecg_name.MEDICAL_RECORD_NUMBER.str.contains('BI|SLR'))]
    ecg_name = ecg_name[(ecg_name.acquisitiondate < '2022-01-01') | (ecg_name.acquisitiondate > '1950-01-01')]
    ecg_name = ecg_name.sort_values(by=['acquisitiondate'])
    ecg_name.acquisitiondate = pd.to_datetime(ecg_name.acquisitiondate)

    # load ehr data for patients with cardiovascular ICD
    # number of iterations for 2.3GB file ^ == 4 iterations ~ 2 minutes
    size_of_chunk = 10 ** 7
    cardio_dx = pd.DataFrame()
    for chunk in pd.read_csv(utils.ehr_cihd_fn, delimiter='\t', chunksize=size_of_chunk):
        chunk = chunk.astype({'MEDICAL_RECORD_NUMBER': str})
        chunk = chunk.drop_duplicates()
        cardio_dx = cardio_dx.append(chunk)

    # process and filter ehr data
    cardio_dx = cardio_dx[cardio_dx['DATE_OF_BIRTH'] != '0000-00-00 00:00:00']
    cardio_dx = cardio_dx[cardio_dx['CALENDAR_DATE'] != '0000-00-00 00:00:00']
    cardio_dx = cardio_dx[cardio_dx['MEDICAL_RECORD_NUMBER'] != 'MSDW_UNKNOWN']
    cardio_dx = cardio_dx[~(cardio_dx.CONTEXT_DIAGNOSIS_CODE.str.contains('IMO'))]
    cardio_dx.GENDER = cardio_dx.GENDER.str.upper()
    cardio_dx.RACE = cardio_dx.RACE.str.upper()
    cardio_dx = cardio_dx[cardio_dx.GENDER.str.contains('MALE')]
    cardio_dx = cardio_dx[~cardio_dx.CALENDAR_DATE.str.contains('21..\-', regex=True)]
    cardio_dx = cardio_dx[~cardio_dx.CALENDAR_DATE.str.contains('25..\-', regex=True)]
    cardio_dx.DATE_OF_BIRTH = pd.to_datetime(cardio_dx.DATE_OF_BIRTH)
    cardio_dx.CALENDAR_DATE = pd.to_datetime(cardio_dx.CALENDAR_DATE)
    cardio_dx = cardio_dx[(cardio_dx.CALENDAR_DATE < '2020-04-01') | (cardio_dx.CALENDAR_DATE > '1950-01-01')]

    # get control ecg data
    # controls_putative = ecg_name[~(ecg_name.MEDICAL_RECORD_NUMBER.isin(cardio_dx.MEDICAL_RECORD_NUMBER))]
    #controls_putative.MEDICAL_RECORD_NUMBER.drop_duplicates().to_csv(utils.control_putative_fn, header=False, index=False)

    #with gzip.open(utils.df_control_putative_fn, 'wb') as f:
    #    pkl.dump(controls_putative, f)

    # get race and ethnicity groups for cardio dx cohort
    ethnicity_map = mapping.get_ethnicity_map(cardio_dx, 'case')
    race_map = mapping.get_race_map(cardio_dx, 'case')
    race_ethnicity_map = mapping.get_race_ethnic_groups(ethnicity_map, race_map, cardio_dx)
    cardio_dx_re = pd.merge(cardio_dx, race_ethnicity_map, on='MEDICAL_RECORD_NUMBER', how='left')

    # get meta-group which is gender + race/ethnicity
    cardio_dx_re['META_GROUP'] = cardio_dx_re.GENDER.str.upper() + '_' + cardio_dx_re.GROUP_RACE_ETHNICITY

    # get chronic ischemic heart disease patients
    cad_dx = cardio_dx_re[cardio_dx_re.CONTEXT_DIAGNOSIS_CODE.str.contains('I25.1')]
    cad_dx = cad_dx.sort_values(by=['CALENDAR_DATE'])
    cad_dx = cad_dx.drop_duplicates(subset=['MEDICAL_RECORD_NUMBER'], keep='first')

    #  get case dataset of ehr and ecg data combination
    cases = pd.merge(ecg_name, cad_dx, on='MEDICAL_RECORD_NUMBER', how='inner')
    cases.loc[:, 'label'] = '1'
    cases = cases.reset_index(drop=True)
    cases = cases[(cases['age'] >= 20) & (cases['age'] <= 90)]
    cases['full_path'] = cases.path + '/' + cases.filename
    cases.full_path = cases.full_path.str.replace('/sharepoint/ecg/', '/sc/arion/projects/mscic1/data/GE_ECG_RSYNC/')

    # time component
    cases['time_delta'] = cases.acquisitiondate - cases.CALENDAR_DATE
    cases['years_icd_ecg'] = cases.time_delta.dt.days / 365

    # create age binned variable
    age_bins = list(range(20, 101, 5))
    cases['age_binned'] = pd.cut(cases['age'], age_bins)

    # if group == 't_after':
    #     cases = cases.loc[cases.time_delta.dt.days >= 0]
    # if group == 't_before':
    #     cases = cases.loc[cases.time_delta.dt.days < 0]

    # groups = ['t_plus_5', 't_plus_1', 't_minus_1', 't_minus_5', 't_minus_10']

    # for g in groups:

    if g == 't_plus_5':
        sub_case = cases.loc[(cases.time_delta.dt.days > 365) & (cases.time_delta.dt.days <= 1825)]
    if g == 't_plus_1':
        sub_case = cases.loc[(cases.time_delta.dt.days > 0) & (cases.time_delta.dt.days <= 365)]
    if g == 't_minus_1':
        sub_case = cases.loc[(cases.time_delta.dt.days > -365) & (cases.time_delta.dt.days <= 0)]
    if g == 't_minus_5':
        sub_case = cases.loc[(cases.time_delta.dt.days > -1825) & (cases.time_delta.dt.days <= -365)]
    if g == 't_minus_10':
        sub_case = cases.loc[(cases.time_delta.dt.days > -3650) & (cases.time_delta.dt.days <= -1825)]
    # save cases dataframe
    data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
    fn = data_dir + 'cases_' + g + '_demographics_20221102.pkl.gz'
    with gzip.open(fn, 'wb') as f:
        pkl.dump(sub_case, f)
    case_measurements, case_demog = process.get_ecg_measurements(sub_case)
    measurement_df = pd.DataFrame.from_records(case_measurements)
    cases_df = pd.concat([case_demog.reset_index(drop=True), measurement_df], axis=1)
    fn = data_dir + 'cases_' + g + '_measurement-demographics_20221102.pkl.gz'
    with gzip.open(fn, 'wb') as f:
        pkl.dump(cases_df, f)

    demographics = cases_df[
        ['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate', 'MEDICAL_RECORD_NUMBER',
         'PERSON_KEY', 'GENDER', 'RACE', 'PATIENT_ETHNIC_GROUP', 'DATE_OF_BIRTH', 'CALENDAR_DATE',
         'GROUP_RACE_ETHNICITY', 'META_GROUP', 'full_path', 'time_delta', 'years_icd_ecg',
         'age_binned', 'CONTEXT_DIAGNOSIS_CODE']]
    # variables to include in the model
    data = cases_df[
        ['label', 'VentricularRate', 'AtrialRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
         'PAxis', 'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset',
         'ECGSampleBase', 'ECGSampleExponent', 'QTcFrederica', 'SystolicBP', 'DiastolicBP']]

    data = data.astype(float)
    data['label'] = data['label'].astype(int)

    variables_dropped = ['SystolicBP', 'DiastolicBP', 'QTcFrederica', 'ECGSampleBase', 'ECGSampleExponent']
    measurement_df = data.drop(variables_dropped, axis='columns')

    cohort_nan = measurement_df.isna().sum(axis='columns')
    # cohort_nan.value_counts()
    # remove samples that have more than 5 out of 15 variables missing
    cohort_impute = cohort_nan[cohort_nan <= 5]

    # Drop demographic rows to match
    measurement_df = measurement_df.loc[cohort_impute.index]
    demographics_df = demographics.loc[cohort_impute.index]

    cases_split_df = pd.concat([demographics_df, measurement_df], axis=1)

    cases_split_df.drop_duplicates(subset=['MEDICAL_RECORD_NUMBER'], inplace=True)

    fn = data_dir + 'cases_split_' + g + '_measurement-demographics_20221102.pkl.gz'
    with gzip.open(fn, 'wb') as f:
        pkl.dump(cases_split_df, f)

    # # create icd-ecg timedelta binned variable
    # three_year_bin = list(range(-39, 46, 3))
    # cases['three_year_binned'] = pd.cut(cases['years_icd_ecg'], three_year_bin)
    # timedelta_bins = cases.three_year_binned.drop_duplicates()
    # cases_before = cases.loc[cases.time_delta.dt.days < 0]
    # cases_after = cases.loc[cases.time_delta.dt.days >= 0]
    # need to do! make sure only MRNs that have 1 ecg before and after WITHOUT accounting for window
    # cases = cases.loc[cases.MEDICAL_RECORD_NUMBER.isin(cases_before.MEDICAL_RECORD_NUMBER) & cases.MEDICAL_RECORD_NUMBER.isin(cases_after.MEDICAL_RECORD_NUMBER)]
    # cases_subset = cases.loc[(cases.time_delta.dt.days >= time_start) & (cases.time_delta.dt.days <= time_stop)]
