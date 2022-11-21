import sys
import pandas as pd
import numpy as np
import utils
sys.path.insert(0, '/sc/arion/projects/mscic1/cad_ecg/cardio_phenotyping/code/cohort_selection/')


def get_ethnicity_map(ehr_df, cohort_group):
    # load the map file
    ethnicity_map = pd.read_csv(utils.ethnicity_fn)

    # get the values from the ehr dataframe
    ethnic_opt = ehr_df.PATIENT_ETHNIC_GROUP.drop_duplicates()
    ethnic_opt = pd.DataFrame(ethnic_opt)
    ethnic_opt['PATIENT_ETHNIC_GROUP'] = ethnic_opt.PATIENT_ETHNIC_GROUP.str.upper()
    ethnic_opt = ethnic_opt.reset_index(drop=True)

    # get the values missing from the map file
    ret = ethnic_opt[~(ethnic_opt.PATIENT_ETHNIC_GROUP.isin(ethnicity_map.CODE))]
    ret.reset_index(drop=True, inplace=True)
    ret.drop_duplicates(inplace=True)
    ret.reset_index(drop=True, inplace=True)
    if cohort_group == 'control':
        result = ret.loc[1:7, :]
        result.reset_index(drop=True, inplace=True)
        group_ethnic = pd.Series(
            ['NON-HISPANIC-LATINO', 'HISPANIC-LATINO', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'])
        ethnic_description = pd.Series(
            ['NON-HISPANIC-LATINO', 'HISPANIC-LATINO', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'])
        result = pd.concat([result, ethnic_description, group_ethnic], axis=1, ignore_index=True)
        result.columns = ['CODE', 'DESCRIPTION', 'GROUP_ETHNIC']
    elif cohort_group == 'case':
        result = ret.loc[1:7, :]
        result.reset_index(drop=True, inplace=True)
        group_ethnic = pd.Series(
            ['UNKNOWN', 'NON-HISPANIC-LATINO', 'HISPANIC-LATINO', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'])
        ethnic_description = pd.Series(
            ['UNKNOWN', 'NON-HISPANIC-LATINO', 'HISPANIC-LATINO', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'])
        result = pd.concat([result, ethnic_description, group_ethnic], axis=1, ignore_index=True)
        result.columns = ['CODE', 'DESCRIPTION', 'GROUP_ETHNIC']

    # concatenate the original map to the additional map
    ethnicity_map = pd.concat([ethnicity_map, result], ignore_index=True)
    ethnicity_map = ethnicity_map.astype({'CODE': str, 'DESCRIPTION': str, 'GROUP_ETHNIC': str})

    return ethnicity_map


def get_race_map(ehr_df, cohort_group):
    # load the map file
    rac_map = pd.read_csv(utils.race_fn)

    # get the values from the ehr dataframe
    race_opt = ehr_df.RACE.drop_duplicates()
    race_opt = pd.DataFrame(race_opt)
    race_opt['RACE'] = race_opt.RACE.str.upper()
    race_opt = race_opt.reset_index(drop=True)

    # get the values missing from the map file
    res = race_opt[~(race_opt.RACE.isin(rac_map.CODE))]
    res.reset_index(drop=True, inplace=True)
    res.drop_duplicates(inplace=True)
    res.reset_index(drop=True, inplace=True)
    result2 = res.loc[1:, :]
    result2.reset_index(drop=True, inplace=True)
    if cohort_group == 'control':
        group_race = pd.Series(['UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'])
        race_description = pd.Series(['UNKNOWN', 'UNKNOWN', 'UNKNOWN', 'UNKNOWN'])
        result2 = pd.concat([result2, race_description, group_race], axis=1, ignore_index=True)
        result2.columns = ['CODE', 'DESCRIPTION', 'GROUP_RACE']

    elif cohort_group == 'case':
        group_race = pd.Series(['UNKNOWN', 'UNKNOWN', 'UNKNOWN'])
        race_description = pd.Series(['UNKNOWN', 'UNKNOWN', 'UNKNOWN'])
        result2 = pd.concat([result2, race_description, group_race], axis=1, ignore_index=True)
        result2.columns = ['CODE', 'DESCRIPTION', 'GROUP_RACE']

    # concatenate the original map to the additional map
    rac_map = pd.concat([rac_map, result2], ignore_index=True)
    rac_map = rac_map.astype({'CODE': str, 'DESCRIPTION': str, 'GROUP_RACE': str})

    return rac_map


def get_race_ethnic_groups(ethn_map, rac_map, ehr_df):

    ethnic_df = pd.merge(ehr_df, ethn_map, left_on='PATIENT_ETHNIC_GROUP', right_on='CODE', how='left')
    ethnic_df = ethnic_df.loc[:, ['MEDICAL_RECORD_NUMBER', 'GROUP_ETHNIC']]
    ethnic_df['val'] = 1
    ethnic_df.sort_values(by=['MEDICAL_RECORD_NUMBER'], inplace=True, ignore_index=True)


    ethnic_pv = pd.pivot_table(ethnic_df, values='val', index=['MEDICAL_RECORD_NUMBER'],
                               columns=['GROUP_ETHNIC'], aggfunc=np.sum, fill_value=0)
    ethnic_pv.reset_index(inplace=True)
    ethnic_mt = pd.melt(ethnic_pv,
                        id_vars=['MEDICAL_RECORD_NUMBER'],
                        value_vars=['HISPANIC-LATINO', 'NON-HISPANIC-LATINO', 'UNKNOWN'])
    ethnic_mt.sort_values(by=['MEDICAL_RECORD_NUMBER', 'value'], inplace=True)
    ethnic_group = ethnic_mt.drop_duplicates('MEDICAL_RECORD_NUMBER', keep='last')
    group_hl = ethnic_group.loc[ethnic_group.GROUP_ETHNIC == 'HISPANIC-LATINO']

    #  what if it is a tie -___-
    # set these non-HL/unknown to continue on with race identification
    # auto set to unknown because of alphabetical order of category values

    # what if hispanic-latino is max but count is zero
    # would not be included in the pivot table to begin with

    # what max category is tied with another category?
    #  HL + UNKNOWN auto set to UNKNOWN because of alphabetical order when all three are equal
    #  HL + non-HL auto set to non-HL which effective becomes unknown bc then goes to race ID
    #print(ehr_df.columns)
    race_df = pd.merge(ehr_df, rac_map, left_on='RACE', right_on='CODE', how='left')
    race_df = race_df.loc[:, ['MEDICAL_RECORD_NUMBER', 'GROUP_RACE']]
    race_df['val'] = 1
    race_df.sort_values(by=['MEDICAL_RECORD_NUMBER'], inplace=True, ignore_index=True)

    race_pv = pd.pivot_table(race_df, values='val', index=['MEDICAL_RECORD_NUMBER'],
                             columns=['GROUP_RACE'], aggfunc=np.sum, fill_value=0)
    race_pv.reset_index(inplace=True)

    race_mt = pd.melt(race_pv,
                      id_vars=['MEDICAL_RECORD_NUMBER'],
                      value_vars=['AMERICAN-INDIAN-OR-ALASKA-NATIVE', 'ASIAN', 'BLACK-OR-AFRICAN-AMERICAN',
                                  'CAUCASIAN-OR-WHITE', 'HISPANIC-LATINO', 'NATIVE-HAWAIIAN-OR-PACIFIC-ISLANDER',
                                  'OTHER', 'UNKNOWN'])
    race_mt.sort_values(by=['MEDICAL_RECORD_NUMBER', 'value'], inplace=True)
    race_group = race_mt.drop_duplicates('MEDICAL_RECORD_NUMBER', keep='last')


    race_ethnicity = race_group[~(race_group.MEDICAL_RECORD_NUMBER.isin(group_hl.MEDICAL_RECORD_NUMBER))]
    race_ethnicity = race_ethnicity.loc[:, ['MEDICAL_RECORD_NUMBER', 'GROUP_RACE']]
    race_ethnicity.columns = ['MEDICAL_RECORD_NUMBER', 'GROUP_RACE_ETHNICITY']

    group_hl = group_hl.loc[:, ['MEDICAL_RECORD_NUMBER', 'GROUP_ETHNIC']]
    group_hl.columns = ['MEDICAL_RECORD_NUMBER', 'GROUP_RACE_ETHNICITY']

    all_race_ethnicity = pd.concat([race_ethnicity, group_hl], ignore_index=True)

    missing = ehr_df[~(ehr_df.MEDICAL_RECORD_NUMBER.isin(all_race_ethnicity.MEDICAL_RECORD_NUMBER))]
    missing_mrn = missing.loc[:, ['MEDICAL_RECORD_NUMBER']]
    missing_mrn['GROUP_RACE_ETHNICITY'] = 'UNKNOWN'
    missing_mrn.drop_duplicates(subset='MEDICAL_RECORD_NUMBER', inplace=True)

    race_ethnicity_dt = pd.concat([all_race_ethnicity, missing_mrn], ignore_index=True)

    return race_ethnicity_dt
