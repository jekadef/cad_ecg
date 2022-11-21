import sys
import pandas as pd
import pickle as pkl
import gzip
import argparse
sys.path.insert(0, '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/code/baseline_models')
import utils
import preprocess
import split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, PredefinedSplit, GroupShuffleSplit, StratifiedShuffleSplit
# sys.path.insert(0, '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/code/cohort_selection')

data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
groups = ['t_plus_1', 't_minus_1', 't_minus_5', 't_minus_10']
sets = ['temp', 'test']
cohort_dict = {}
for g in groups:
    fn = data_dir + 'cohort_' + g + '_input_dict_20221102.pkl.gz'
    with gzip.open(fn, 'rb') as f:
        input_dict = pkl.load(f)
    cohort_dict[g] = input_dict['demog_dict']

# time to ECG from diagnosis date
for g in groups:
    for s in sets:
        df = cohort_dict[g][s]
        print(str(g))
        print(str(s))
        cases = df[df.label == 1]
        print('mean: ' + str(round(cases.years_icd_ecg.mean() * 365, 2)))
        print('sd: ' + str(round(cases.years_icd_ecg.std() * 365, 3)))
        print('median: ' + str(round(cases.years_icd_ecg.median() * 365, 2)))
        print('range: ' + str(round(cases.years_icd_ecg.min() * 365, 2)) + ' to ' + str(round(cases.years_icd_ecg.max() * 365, 2)))


# number of cases and controls
for g in groups:
    for s in sets:
        df = cohort_dict[g][s]
        print(str(g))
        print(str(s))
        print('cases ' + str(len(df[df.label == 1])))
        print('controls ' + str(len(df[df.label == 0])))

# age
for g in groups:
    for s in sets:
        df = cohort_dict[g][s]
        print(str(g))
        print(str(s))
        cases = df[df.label == 1]
        print('mean: ' + str(round(cases.age.mean(), 2)))
        print('sd: ' + str(round(cases.age.std(), 3)))
        print('median: ' + str(round(cases.age.median(), 2)))
        print('range: ' + str(round(cases.age.min(), 2)) + ' to ' + str(round(cases.age.max(), 2)))

full_cohort = pd.DataFrame()
for g in groups:
    for s in sets:
        df = cohort_dict[g][s]
        full_cohort = pd.concat([full_cohort, df])

race_ethnicity = ['MALE_CAUCASIAN-OR-WHITE', 'FEMALE_CAUCASIAN-OR-WHITE',
                  'MALE_OTHER', 'FEMALE_OTHER',
                  'MALE_HISPANIC-LATINO', 'FEMALE_HISPANIC-LATINO',
                  'MALE_UNKNOWN', 'FEMALE_UNKNOWN',
                  'FEMALE_ASIAN', 'MALE_ASIAN',
                  'FEMALE_BLACK-OR-AFRICAN-AMERICAN', 'MALE_BLACK-OR-AFRICAN-AMERICAN',
                  'MALE_NATIVE-HAWAIIAN-OR-PACIFIC-ISLANDER', 'FEMALE_NATIVE-HAWAIIAN-OR-PACIFIC-ISLANDER',
                  'MALE_AMERICAN-INDIAN-OR-ALASKA-NATIVE', 'FEMALE_AMERICAN-INDIAN-OR-ALASKA-NATIVE']

for g in groups:
    print(str(g))
    for s in sets:
        print(str(s))
        df = cohort_dict[g][s]
        for r in race_ethnicity:
            print(r)
            round(len(df[df.META_GROUP == r]) / len(df) * 100, 2)


for r in race_ethnicity:
    case = full_cohort[full_cohort.label == 1]
    control = full_cohort[full_cohort.label == 0]
    perc_control = len(control[control.META_GROUP == r]) / len(control) * 100
    perc_case = len(case[case.META_GROUP == r]) / len(case) * 100
    print(str(r))
    print('case: ' + str(round(perc_case, 2)))
    print('control: ' + str(round(perc_control, 2)))

datdir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/bestmodel_results/'
models = ['random_forest', 'logistic_regression']
for m in models:
    for g in groups:
        fn = datdir + m + '_' + g + '_best_model_performance_20221102.pkl.gz'
        with gzip.open(fn, 'rb') as f:
            res = pkl.load(f)
            print(res)

models = ['logistic_regression', 'random_forest']
groups = ['t_plus_1', 't_minus_1', 't_minus_5', 't_minus_10']

dat_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/baseline_results/'
fn = dat_dir + 'gridsearch_results/' + m + '_' + g + '_grid_results_20221102.pkl.gz'

with gzip.open(fn, 'rb') as f:
    res = pkl.load(f)
    print(res)


cases_df = full_cohort[full_cohort.label == 1]
cases_df.reset_index(inplace=True)

control_df = full_cohort[full_cohort.label == 0]
len(control_df.MEDICAL_RECORD_NUMBER.unique())

pn = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/plots/years_icd_ecg_20221102.pdf'
sns.histplot(data=cases_df, x="years_icd_ecg", binwidth=0.2)
plt.savefig(pn)
plt.close()