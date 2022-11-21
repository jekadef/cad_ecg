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
    parser.add_argument('--afterdata', type=str, required=True)
    args = parser.parse_args()

    after_set = args.afterdata

    data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'

    control_fn = data_dir + 'controls_split_' + after_set + '_measurement-demographics_20221102.pkl.gz'
    cases_fn = data_dir + 'cases_split_' + after_set + '_measurement-demographics_20221102.pkl.gz'
    cohort_fn = data_dir + 'cohort_split_' + after_set + '_measurement-demographics_20221102.pkl.gz'

    with gzip.open(cases_fn, 'rb') as f:
        case = pkl.load(f)

    with gzip.open(control_fn, 'rb') as f:
        control = pkl.load(f)

    cohort = pd.concat([case, control], axis=0, ignore_index=True)
    cohort.reset_index(inplace=True, drop=True)

    demographics = cohort[['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate', 'MEDICAL_RECORD_NUMBER',
                    'PERSON_KEY', 'GENDER', 'RACE', 'PATIENT_ETHNIC_GROUP', 'DATE_OF_BIRTH', 'CALENDAR_DATE',
                    'GROUP_RACE_ETHNICITY', 'META_GROUP', 'full_path', 'time_delta', 'years_icd_ecg',
                    'age_binned', 'weights', 'CONTEXT_DIAGNOSIS_CODE']]

    # variables to include in the model
    data = cohort[['label', 'VentricularRate', 'AtrialRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
                    'PAxis', 'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset']]
    data = data.astype(float)
    data['label'] = data['label'].astype(int)

    # Drop out of range values #
    cohort_data_df = preprocess.quantile_filter(data, 0.0005, 0.9995)

    # Drop demographic rows to match
    demographics_df = demographics.loc[cohort_data_df.index]
    dataset = pd.concat([cohort_data_df, demographics_df], axis=1)

    # Even out cases and controls
    case = dataset[dataset.label == 1]
    control = dataset[dataset.label == 0]

    # new_control = control.sample(n=len(case), random_state=8)
    new_control = control.sample(n=len(case), random_state=10)
    dataset = pd.concat([case, new_control], axis=0, ignore_index=True)
    dataset.reset_index(inplace=True, drop=True)

    data_dict, group_split_dict, demog_dict = split.get_sets(dataset)

    data_imputer, data_dict['Xtrain'] = preprocess.get_imputed(data_dict['Xtrain'], utils.imputation_type)
    data_dict['Xval'] = data_imputer.fit_transform(data_dict['Xval'])
    data_dict['Xtest'] = data_imputer.fit_transform(data_dict['Xtest'])
    data_dict['Xtemp'] = data_imputer.fit_transform(data_dict['Xtemp'])
    # Scale the data #
    scaling, data_dict['Xtrain'] = preprocess.get_scaling(data_dict['Xtrain'], utils.scaling_type)
    data_dict['Xval'] = scaling.transform(data_dict['Xval'])
    data_dict['Xtest'] = scaling.transform(data_dict['Xtest'])
    data_dict['Xtemp'] = scaling.transform(data_dict['Xtemp'])

    input_dict = {}
    input_dict['data_dict'] = data_dict
    input_dict['demog_dict'] = demog_dict

    # dont save as seperate files
    fn = data_dir + 'cohort_' + after_set + '_input_dict_20221102.pkl.gz'
    with gzip.open(fn, 'wb') as f:
        pkl.dump(input_dict, f)

