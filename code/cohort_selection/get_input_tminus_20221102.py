import sys
import pandas as pd
import pickle as pkl
import gzip
import argparse
sys.path.insert(0, '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/code/baseline_models')
import utils
import preprocess
import split
from sklearn.model_selection import train_test_split, PredefinedSplit, GroupShuffleSplit, StratifiedShuffleSplit
# sys.path.insert(0, '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/code/cohort_selection')
#
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--beforedata', type=str, required=True)
    parser.add_argument('--afterdata', type=str, required=True)
    args = parser.parse_args()

    after_set = args.afterdata
    before_set = args.beforedata

    data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'

    # dont save as seperate files
    fn = data_dir + 'cohort_' + after_set + '_input_dict_20221102.pkl.gz'
    with gzip.open(fn, 'rb') as f:
        input_dict = pkl.load(f)

    test_df = input_dict['demog_dict']['test']
    # test_df[test_df.label == 1]
    # test_df[test_df.label == 0]

    # before_groups = ['t_minus_1', 't_minus_5', 't_minus_10']

    control_fn = data_dir + 'controls_split_' + before_set + '_measurement-demographics_20221102.pkl.gz'
    cases_fn = data_dir + 'cases_split_' + before_set + '_measurement-demographics_20221102.pkl.gz'
    cohort_fn = data_dir + 'cohort_split_' + before_set + '_measurement-demographics_20221102.pkl.gz'

    with gzip.open(cases_fn, 'rb') as f:
        case = pkl.load(f)

    with gzip.open(control_fn, 'rb') as f:
        control = pkl.load(f)

    cohort = pd.concat([case, control], axis=0, ignore_index=True)
    cohort.reset_index(inplace=True, drop=True)

    demographics = cohort[
        ['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate', 'MEDICAL_RECORD_NUMBER',
         'PERSON_KEY', 'GENDER', 'RACE', 'PATIENT_ETHNIC_GROUP', 'DATE_OF_BIRTH', 'CALENDAR_DATE',
         'GROUP_RACE_ETHNICITY', 'META_GROUP', 'full_path', 'time_delta', 'years_icd_ecg',
         'age_binned', 'weights', 'CONTEXT_DIAGNOSIS_CODE']]

    # variables to include in the model
    data = cohort[
        ['label', 'VentricularRate', 'AtrialRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
         'PAxis', 'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset']]
    data = data.astype(float)
    data['label'] = data['label'].astype(int)

    # Drop out of range values #
    cohort_data_df = preprocess.quantile_filter(data, 0.0005, 0.9995)

    # Drop demographic rows to match
    demographics_df = demographics.loc[cohort_data_df.index]
    dataset = pd.concat([cohort_data_df, demographics_df], axis=1)

    num_case = len(dataset[dataset.label == 1])
    num_cohort = num_case * 2
    num_test_needed = int(num_cohort * 0.2)
    num_temp_needed = int(num_cohort * 0.8)

    # number of cases 50,764
    # number of controls needed 50,764 --> have 89,684

    # number of temp needed 81,222
    # number of each temp case/controls needed 40,611 --> have 43,894/82,655

    # number of test controls needed 20,305
    # number of each test case/controls needed 10,152 --> have 6,872/7,031

    # number of total cohort 101,528

    num_case_test = int(num_test_needed / 2) # number of cases needed in test
    num_case_temp = int(num_temp_needed / 2) # number of controls needd in test

    after_case_mrn = test_df[test_df.label == 1].MEDICAL_RECORD_NUMBER
    after_control_mrn = test_df[test_df.label == 0].MEDICAL_RECORD_NUMBER

    case2 = dataset[dataset.label == 1]
    control2 = dataset[dataset.label == 0]

    test_case = case2[case2.MEDICAL_RECORD_NUMBER.isin(after_case_mrn)]
    test_control = control2[control2.MEDICAL_RECORD_NUMBER.isin(after_control_mrn)]
    temp_case = case2[~case2.MEDICAL_RECORD_NUMBER.isin(after_case_mrn)]
    temp_control = control2[~control2.MEDICAL_RECORD_NUMBER.isin(after_control_mrn)]

    test_case_add = num_case_test - len(test_case)
    test_control_add = num_case_test - len(test_control)

    # add or subtract some samples to reach 20% test set size
    ## if positive then take some from train+dev
    if test_case_add > 0:
        case_add_df = temp_case.sample(n=test_case_add, random_state=10)
        temp_case = temp_case.loc[~temp_case.index.isin(case_add_df.index)]
        test_case = pd.concat([test_case, case_add_df])
    if test_control_add > 0:
        control_add_df = temp_control.sample(n=test_control_add, random_state=10)
        temp_control = temp_control.loc[~temp_control.index.isin(control_add_df.index)]
        test_control = pd.concat([test_control, control_add_df])

    ### if negative then subsample from test
    if test_case_add < 0:
        test_case = test_case.sample(n=abs(test_case_add), random_state=10)
    if test_control_add < 0:
        test_control = test_control.sample(n=abs(test_control_add), random_state=10)

    temp_case_add = num_case_temp - len(temp_case)
    temp_control_add = num_case_temp - len(temp_control)

    if temp_case_add < 0:
        temp_case = temp_case.sample(n=num_case_temp, random_state=10)
    if temp_control_add < 0:
        temp_control = temp_control.sample(n=abs(num_case_temp), random_state=10)

    measurement_vars = ['VentricularRate', 'AtrialRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
                         'PAxis', 'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset']
    demographic_vars = ['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate', 'MEDICAL_RECORD_NUMBER',
                        'PERSON_KEY', 'GENDER', 'RACE', 'PATIENT_ETHNIC_GROUP', 'DATE_OF_BIRTH', 'CALENDAR_DATE',
                        'GROUP_RACE_ETHNICITY', 'META_GROUP', 'full_path', 'label', 'time_delta', 'years_icd_ecg',
                        'age_binned', 'weights', 'CONTEXT_DIAGNOSIS_CODE']

    data_dict = {}
    demog_dict = {}
    res = pd.concat([test_case, test_control])
    data_dict['ytest'] = res.loc[:, 'label'].to_numpy()
    data_dict['Xtest'] = res.loc[:, measurement_vars].to_numpy()
    demog_dict['test'] = res.loc[:, demographic_vars]

    res  = pd.concat([temp_case, temp_control])
    data_dict['ytemp'] = res.loc[:, 'label'].to_numpy()
    data_dict['Xtemp'] = res.loc[:, measurement_vars].to_numpy()
    demog_dict['temp'] = res.loc[:, demographic_vars]
    # dataset[dataset.patientid.isin(group_split_dict['temp'])].loc[:, 'patientid']

    x_temp = res.loc[:, measurement_vars]
    y_temp = res.loc[:, 'label']
    demog_temp = res.loc[:, demographic_vars]

    sss = StratifiedShuffleSplit(n_splits=1, train_size=.75, random_state=8)
    for train_ix, val_ix in sss.split(x_temp, y_temp):
        split_index = [-1 if x in train_ix else 0 for x in y_temp.index]
        ps = PredefinedSplit(test_fold=split_index)
        data_dict['predef_split'] = ps
        data_dict['Xtrain'] = x_temp.iloc[train_ix].to_numpy()
        data_dict['ytrain'] = y_temp.iloc[train_ix].to_numpy()
        # group_split['train'] = group_temp.iloc[train_ix]
        demog_dict['train'] = demog_temp.iloc[train_ix]
        data_dict['Xval'] = x_temp.iloc[val_ix].to_numpy()
        data_dict['yval'] = y_temp.iloc[val_ix].to_numpy()
        # group_split['val'] = group_temp.iloc[val_ix]
        demog_dict['val'] = demog_temp.iloc[val_ix]

    data_imputer, data_dict['Xtrain'] = preprocess.get_imputed(data_dict['Xtrain'], utils.imputation_type)
    data_dict['Xval'] = data_imputer.fit_transform(data_dict['Xval'])
    data_dict['Xtest'] = data_imputer.fit_transform(data_dict['Xtest'])
    data_dict['Xtemp'] = data_imputer.fit_transform(data_dict['Xtemp'])
    # Scale the data #
    scaling, data_dict['Xtrain'] = preprocess.get_scaling(data_dict['Xtrain'], utils.scaling_type)
    data_dict['Xval'] = scaling.transform(data_dict['Xval'])
    data_dict['Xtest'] = scaling.transform(data_dict['Xtest'])
    data_dict['Xtemp'] = scaling.transform(data_dict['Xtemp'])
    # Get groups by index
    # train_demo = demographics_df.loc[train_idx]
    # val_demo = demographics_df.loc[val_idx]
    # test_demo = demographics_df.loc[test_idx]
    # save demographics dict one for each data configuration ie N=9
    # demographics_dict = {'training': train_demo, 'validation': val_demo, 'test': test_demo}

    b_input_dict = {}
    b_input_dict['data_dict'] = data_dict
    b_input_dict['demog_dict'] = demog_dict

    # dont save as seperate files
    fn = data_dir + 'cohort_' + before_set + '_input_dict_20221102.pkl.gz'
    with gzip.open(fn, 'wb') as f:
        pkl.dump(b_input_dict, f)
