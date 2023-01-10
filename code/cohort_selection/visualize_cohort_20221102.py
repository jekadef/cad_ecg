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


missing_cases = new_full_cohort[~(new_full_cohort.MEDICAL_RECORD_NUMBER.isin(old_post_df.MRN))]
missing_cases = new_full_cohort[~(new_full_cohort.MEDICAL_RECORD_NUMBER.isin(old_post_df.MRN))]
full_case_post
new_cohort_df2.MRN
##

groups = ['t_plus_1', 't_minus_1', 't_minus_5', 't_minus_10']
# control_fn = data_dir + 'controls_split_' + time_group + '_measurement-demographics_20221102.pkl.gz'
full_case_post = pd.DataFrame()
full_case_pre = pd.DataFrame()
for time_group in groups:
    data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
    cases_fn = data_dir + 'cases_split_' + time_group + '_measurement-demographics_20221102.pkl.gz'
    with gzip.open(cases_fn, 'rb') as f:
        case = pkl.load(f)
    full_case_post = pd.concat([full_case_post, case])
    # print('case after processing')
    # print(len(case.MEDICAL_RECORD_NUMBER.unique()))
    data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
    fn2 = data_dir + 'cases_' + time_group + '_demographics_20221102.pkl.gz'
    with gzip.open(fn2, 'rb') as f:
        case_new_before_process = pkl.load(f)
    full_case_pre = pd.concat([full_case_pre, case_new_before_process])
    # print('case before processing')
    # print(len(case_new_before_process.MEDICAL_RECORD_NUMBER.unique()))

print(len(full_case_pre.MEDICAL_RECORD_NUMBER.unique()))
print(len(full_case_post.MEDICAL_RECORD_NUMBER.unique()))

#old dataset
full_case_pre_old = pd.DataFrame()

data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/cohort_data/'
fn = data_dir + 'cohort_split_t_after_measurement-demographics_20220912.pkl.gz'

with gzip.open(fn, 'rb') as f:
    after_cohort_dict = pkl.load(f)

old_dataset = after_cohort_dict['from0to1']
old_case = old_dataset[old_dataset.label == 1]

full_case_pre_old = pd.concat([full_case_pre_old, old_case])
# len(old_case.MEDICAL_RECORD_NUMBER.unique())

data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/cohort_data/'
fn = data_dir + 'cohort_split_t_before_measurement-demographics_20220912.pkl.gz'
with gzip.open(fn, 'rb') as f:
    before_cohort_dict = pkl.load(f)

groups = ['from-1to0', 'from-5to-1', 'from-10to-5']

for g in groups:
    old_dataset = before_cohort_dict[g]
    old_case = old_dataset[old_dataset.label == 1]
    full_case_pre_old = pd.concat([full_case_pre_old, old_case])
    # print(g)

print(len(full_case_pre_old.MEDICAL_RECORD_NUMBER.unique()))



###
data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/'
groups = ['t_plus_1', 't_minus_1', 't_minus_5', 't_minus_10']

new_cohort_dict = {}
new_cohort_dict_data = {}
for g in groups:
    fn = data_dir + 'cohort_' + g + '_input_dict_20221102.pkl.gz'
    with gzip.open(fn, 'rb') as f:
        input_dict = pkl.load(f)
    new_cohort_dict[g] = input_dict['demog_dict']
    new_cohort_dict_data[g] = input_dict['data_dict']


for k, v in new_cohort_dict.items():
    print(str(k))
    for kk, vv in v.items():
        print(str(kk))
        print('case')
        print(len(vv[vv.label == 1]))
        print('control')
        print(len(vv[vv.label == 0]))

new_cohort_df = pd.DataFrame()
for k, v in new_cohort_dict.items():
    for kk, vv in v.items():
        d = {'label': vv.label, 'MRN': vv.MEDICAL_RECORD_NUMBER}
        df = pd.DataFrame(data=d)
        df['set'] = kk
        df['group'] = k
        new_cohort_df = pd.concat([new_cohort_df, df])

# cohort_df[cohort_df != 'temp']
new_cohort_df2 = new_cohort_df[new_cohort_df.set != 'temp']
new_cohort_df2.reset_index(inplace=True, drop=True)

# pn = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/plots/group-label-hist.pdf'
# sns.histplot(data=new_cohort_df2, multiple='dodge', x='group', hue='label')
# plt.savefig(pn)
# plt.close()
#
# for g in groups:
#     pn = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/plots/set-label-hist-' + str(g) + '.pdf'
#     res_df = new_cohort_df2[new_cohort_df2.group == g]
#     sns.histplot(data=res_df, multiple='dodge', x='set', hue='label')
#     plt.savefig(pn)
#     plt.close()

groups = ['t_plus_1', 't_minus_1', 't_minus_5', 't_minus_10']
sets = ['train', 'val', 'test']
new_full_cohort = pd.DataFrame()
for g in groups:
    for s in sets:
        df = new_cohort_dict[g][s]
        new_full_cohort = pd.concat([new_full_cohort, df])


# old dataset
data_dir = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/cohort_data/'
groups = ['t_after_from0to1', 't_before_from-1to0', 't_before_from-5to-1', 't_before_from-10to-5']

old_cohort_dict = {}
for g in groups:
    fn = data_dir + 'cohort_' + g + '_input_dict_20220912.pkl.gz'
    with gzip.open(fn, 'rb') as f:
        input_dict = pkl.load(f)
    old_cohort_dict[g] = input_dict['demog_dict']

for k, v in old_cohort_dict.items():
    print(str(k))
    for kk, vv in v.items():
        print(str(kk))
        print('case')
        print(len(vv[vv.label == 1]))
        print('control')
        print(len(vv[vv.label == 0]))

old_cohort_df = pd.DataFrame()
for k, v in old_cohort_dict.items():
    for kk, vv in v.items():
        d = {'label': vv.label, 'MRN': vv.MEDICAL_RECORD_NUMBER}
        df = pd.DataFrame(data=d)
        df['set'] = kk
        df['group'] = k
        old_cohort_df = pd.concat([old_cohort_df, df])

# cohort_df[cohort_df != 'temp']
old_cohort_df2 = old_cohort_df[old_cohort_df.set != 'temp']
old_cohort_df2.reset_index(inplace=True, drop=True)

old_post_df = old_cohort_df2[old_cohort_df2.label == 1]

print(len(old_post_df.MRN.unique()))


pn = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/plots/group-label-hist_old-dataset.pdf'
sns.histplot(data=old_cohort_df2, multiple='dodge', x='group', hue='label')
plt.savefig(pn)
plt.close()

for g in groups:
    pn = '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/data/cohort_selection/plots/set-label-hist-' + str(g) + '-old.pdf'
    res_df = old_cohort_df2[old_cohort_df2.group == g]
    sns.histplot(data=res_df, multiple='dodge', x='set', hue='label')
    plt.savefig(pn)
    plt.close()

# # sets = ['temp', 'test']
sets = ['train', 'val', 'test']
# sets = ['temp', 'test']
# sets = ['train', 'val', 'test']

for k, v in new_cohort_dict.items():
    print(str(k))
    for kk, vv in v.items():
        print(str(kk))
        print('case')
        print(len(vv[vv.label == 1]))
        print('control')
        print(len(vv[vv.label == 0]))

for k, v in old_cohort_dict.items():
    print(str(k))
    for kk, vv in v.items():
        print(str(kk))
        print('case')
        print(len(vv[vv.label == 1]))
        print('control')
        print(len(vv[vv.label == 0]))

# >>> len(old_cohort_dict['t_after_from0to1']['temp'].MEDICAL_RECORD_NUMBER.unique())
# 179979
# NEW controls 49685 cases 49684
# OLD controls 283992 cases 292653 **** NOT SPLIT EQUAL
# >>> len(cohort_dict['t_plus_1']['temp'].MEDICAL_RECORD_NUMBER.unique())
# 99365
groups = ['t_after_from0to1', 't_before_from-1to0', 't_before_from-5to-1', 't_before_from-10to-5']
sets = ['train', 'val', 'test']

for g in groups:
    df = old_cohort_df[old_cohort_df.group == g]
    print(str(g))
    for s in sets:
        res = df[df.set == s]
        print(str(len(res.MRN.unique())) + '  ' + str(s))


groups = ['t_plus_1', 't_minus_1', 't_minus_5', 't_minus_10']
sets = ['train', 'val', 'test']

for g in groups:
    df = new_cohort_df[new_cohort_df.group == g]
    print(str(g))
    for s in sets:
        res = df[df.set == s]
        print(str(len(res.MRN.unique())) + '  ' + str(s))


###########

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