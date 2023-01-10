import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer

# what other imports to include ***

def clean_ecg(fn):
    # load ecg data and add column names
    ecg_name = pd.read_csv(fn, header=None)
    ecg_name.columns = ['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate']
    # clean identifier
    ecg_name['MEDICAL_RECORD_NUMBER'] = ecg_name['patientid'].str.lstrip('0')
    # remove missing for misformatted identifiers
    ecg_name = ecg_name[~(ecg_name.MEDICAL_RECORD_NUMBER.isna())]
    ecg_name = ecg_name[~(ecg_name.MEDICAL_RECORD_NUMBER.str.contains('BI|SLR'))]
    # select ecgs collected between 1950 and 2022
    ecg_name = ecg_name[(ecg_name.acquisitiondate < '2022-01-01') | (ecg_name.acquisitiondate > '1950-01-01')]
    ecg_name = ecg_name.sort_values(by=['acquisitiondate'])
    ecg_name.acquisitiondate = pd.to_datetime(ecg_name.acquisitiondate)
    # remove ecgs from patients less than 20 and greater than 90
    ecg_name = ecg_name[(ecg_name['age'] >= 20) & (ecg_name['age'] <= 90)]
    # fix formatting
    ecg_name['full_path'] = ecg_name.path + '/' + ecg_name.filename
    ecg_name.full_path = ecg_name.full_path.str.replace('/sharepoint/ecg/', '/sc/arion/projects/mscic1/data/GE_ECG_RSYNC/')
    # add additional variables
    ecg_name['time_delta'] = ecg_name.acquisitiondate - ecg_name.CALENDAR_DATE
    ecg_name['years_icd_ecg'] = ecg_name.time_delta.dt.days / 365
    age_bins = list(range(20, 101, 5))
    ecg_name['age_binned'] = pd.cut(ecg_name['age'], age_bins)

    return ecg_name

def clean_msdw(fn):
    # load ehr data for patients with cardiovascular ICD
    # number of iterations for 2.3GB file ^ == 4 iterations ~ 2 minutes
    size_of_chunk = 10 ** 7
    cardio_dx = pd.DataFrame()
    for chunk in pd.read_csv(fn, delimiter='\t', chunksize=size_of_chunk):
        chunk = chunk.astype({'MEDICAL_RECORD_NUMBER': str})
        chunk = chunk.drop_duplicates()
        cardio_dx = cardio_dx.append(chunk)

    # remove observations without birthdate or diagnosis date or identifier
    cardio_dx = cardio_dx[cardio_dx['DATE_OF_BIRTH'] != '0000-00-00 00:00:00']
    cardio_dx = cardio_dx[cardio_dx['CALENDAR_DATE'] != '0000-00-00 00:00:00']
    cardio_dx = cardio_dx[cardio_dx['MEDICAL_RECORD_NUMBER'] != 'MSDW_UNKNOWN']
    # remove misformatted icd codes
    cardio_dx = cardio_dx[~(cardio_dx.CONTEXT_DIAGNOSIS_CODE.str.contains('IMO'))]
    # format columns
    cardio_dx.GENDER = cardio_dx.GENDER.str.upper()
    cardio_dx.RACE = cardio_dx.RACE.str.upper()
    # remove non-female/male observations
    cardio_dx = cardio_dx[cardio_dx.GENDER.str.contains('MALE')]
    # remove misformatted dates
    cardio_dx = cardio_dx[~cardio_dx.CALENDAR_DATE.str.contains('21..\-', regex=True)]
    cardio_dx = cardio_dx[~cardio_dx.CALENDAR_DATE.str.contains('25..\-', regex=True)]
    # format dates
    cardio_dx.DATE_OF_BIRTH = pd.to_datetime(cardio_dx.DATE_OF_BIRTH)
    cardio_dx.CALENDAR_DATE = pd.to_datetime(cardio_dx.CALENDAR_DATE)
    # select ehr observatations between 1950 and 2020

    return cardio_dx

def map_ethnicity(ehr_df, cohort_group):
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

def map_race(ehr_df, cohort_group):
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

def map_groups(ehr_df, group):

    ethn_map = map_ethnicity(ehr_df, group)
    rac_map = map_race(ehr_df, group)

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

    msdw_df2 = pd.merge(ehr_df, race_ethnicity_dt, on='MEDICAL_RECORD_NUMBER', how='left')
    msdw_df2['META_GROUP'] = msdw_df2.GENDER.str.upper() + '_' + msdw_df2.GROUP_RACE_ETHNICITY
    ## get meta-group which is gender + race/ethnicity

    return msdw_df2

def get_cad_df(df):
    # get chronic ischemic heart disease patients
    cad_dx = df[df.CONTEXT_DIAGNOSIS_CODE.str.contains('I25.1')]
    cad_dx = cad_dx.sort_values(by=['CALENDAR_DATE'])
    cad_dx = cad_dx.drop_duplicates(subset=['MEDICAL_RECORD_NUMBER'], keep='first')
    return cad_dx

def select_timeframes(df, group):
    if group == 't_plus_5':
        sub_case = df.loc[(df.time_delta.dt.days > 365) & (df.time_delta.dt.days <= 1825)]
    if group == 't_plus_1':
        sub_case = df.loc[(df.time_delta.dt.days > 0) & (df.time_delta.dt.days <= 365)]
    if group == 't_minus_1':
        sub_case = df.loc[(df.time_delta.dt.days > -365) & (df.time_delta.dt.days <= 0)]
    if group == 't_minus_5':
        sub_case = df.loc[(df.time_delta.dt.days > -1825) & (df.time_delta.dt.days <= -365)]
    if group == 't_minus_10':
        sub_case = df.loc[(df.time_delta.dt.days > -3650) & (df.time_delta.dt.days <= -1825)]
    return sub_case

def load_ecg(filename):
    with open(filename, 'r') as infile:
        try:
            ecg_data = xmltodict.parse(infile.read())
            ecg_data = ecg_data['RestingECG']
        except:  # Blank excepts aren't recommended
            return False
            print("can't load ecg")
    return ecg_data

def get_ecg_measurements(cohort):
    error_idx = []
    case_keep_idx = []
    list_measurements = []
    errors = []

    for idx, row in cohort.iterrows():
        try:
            ecg = load_ecg(row['full_path'])
            rhythms = ecg['Waveform'][1]
            raw_lead_data = rhythms['LeadData']
            all_lead_data = {}
            flats = []
            for this_lead in raw_lead_data:
                lead_name = this_lead['LeadID']
                waveform = this_lead['WaveFormData']
                waveform = base64.b64decode(
                    waveform)  # Base64 strings. This returns a bytes object - Which is 16bit integer in turn
                waveform = np.frombuffer(waveform, dtype='int16')
                # filtered_waveform = butterworth(waveform)
                # filtered_waveform = median_filter(filtered_waveform)
                # all_lead_data[lead_name] = filtered_waveform
                if lead_name == 'I':
                    if np.all(waveform[-500:-1]):
                        error_idx.append(idx)
                    elif not np.all(waveform[-500:-1]):
                        case_keep_idx.append(idx)
                        measurements = ecg['RestingECGMeasurements']
                        list_measurements.append(measurements)
        except:
            print('cant load ecg: ' + row['full_path'])
            errors.append(row['full_path'])

    filtered_cohort = cohort.loc[case_keep_idx]
    list_measurements = pd.DataFrame.from_records(list_measurements)
    out_df = pd.concat([filtered_cohort.reset_index(drop=True), list_measurements], axis=1)
    # filtered_cohort_files = filtered_cohort.loc[:, ['full_path', 'label']]

    # return list_measurements, filtered_cohort, filtered_cohort_files
    return out_df

def drop_vars_obs(df):
    demographics = df[
        ['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate', 'MEDICAL_RECORD_NUMBER',
         'PERSON_KEY', 'GENDER', 'RACE', 'PATIENT_ETHNIC_GROUP', 'DATE_OF_BIRTH', 'CALENDAR_DATE',
         'GROUP_RACE_ETHNICITY', 'META_GROUP', 'full_path', 'time_delta', 'years_icd_ecg',
         'age_binned', 'CONTEXT_DIAGNOSIS_CODE']]
    # variables to include in the model
    data = df[
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
    split_df = pd.concat([demographics_df, measurement_df], axis=1)
    split_df.drop_duplicates(subset=['MEDICAL_RECORD_NUMBER'], inplace=True)
    return split_df

def matched_controls(cases_df, controls):
    group_meta = cases_df.META_GROUP.drop_duplicates()
    # timedelta_bins = filtered_cases_dt.three_year_binned.drop_duplicates()
    # selected_cases = cases_df
    sampled_controls = pd.DataFrame()
    for gm in group_meta:
        # print(cases_to_add)
        g_cases = cases_df[cases_df['META_GROUP'] == gm]
        ctrl_to_sample = int(len(g_cases) * 1.8)  # + cases_to_add
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
        sampled_controls = pd.concat([sampled_controls, y], axis=0)

    return sampled_controls

def quantile_filter(x_data, low, high):
    low_quantile = x_data.quantile(low, axis='rows')
    high_quantile = x_data.quantile(high, axis='rows')

    for col in x_data.columns:
        x_data = x_data.loc[((x_data[col] >= low_quantile[col]) & (x_data[col] <= high_quantile[col])) | x_data[col].isna()]

    return x_data

def separate_data_demog(case, control):
    # vars
    demog_vars = ['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate', 'MEDICAL_RECORD_NUMBER',
                  'PERSON_KEY', 'GENDER', 'RACE', 'PATIENT_ETHNIC_GROUP', 'DATE_OF_BIRTH', 'CALENDAR_DATE',
                  'GROUP_RACE_ETHNICITY', 'META_GROUP', 'full_path', 'time_delta', 'years_icd_ecg',
                  'age_binned', 'weights', 'CONTEXT_DIAGNOSIS_CODE']
    data_vars = ['label', 'VentricularRate', 'AtrialRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
                 'PAxis', 'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset']
    cohort = pd.concat([case, control], axis=0, ignore_index=True)
    cohort.reset_index(inplace=True, drop=True)
    demographics = cohort[demog_vars]  # demographic variables
    ecgdata = cohort[data_vars]  # variables to include in the model
    ecgdata = ecgdata.astype(float)
    ecgdata['label'] = ecgdata['label'].astype(int)
    return ecgdata, demographics

def split_sets_tplus(data_df):
    """Takes the dataset and splits into training, validation, test and predefinted splits"""
    data_sets = {}
    group_split = {}
    demographics = {}
    gss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=8)

    x_data = data_df.loc[:, ['VentricularRate', 'AtrialRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
                             'PAxis', 'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset']]
    y_data = data_df.loc[:, 'label']
    demog = data_df.loc[:, ['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate', 'MEDICAL_RECORD_NUMBER',
                            'PERSON_KEY', 'GENDER', 'RACE', 'PATIENT_ETHNIC_GROUP', 'DATE_OF_BIRTH', 'CALENDAR_DATE',
                            'GROUP_RACE_ETHNICITY', 'META_GROUP', 'full_path', 'label', 'time_delta', 'years_icd_ecg',
                            'age_binned', 'weights', 'CONTEXT_DIAGNOSIS_CODE']]

    for temp_ix, test_ix in gss1.split(x_data, y_data):

        data_sets['Xtest'] = x_data.iloc[test_ix].to_numpy()
        data_sets['ytest'] = y_data.iloc[test_ix].to_numpy()
        # group_split['test'] = group.iloc[test_ix]
        demographics['test'] = demog.iloc[test_ix]

        x_temp = x_data.iloc[temp_ix]
        y_temp = y_data.iloc[temp_ix]
        # group_temp = group.iloc[temp_ix]
        demog_temp = demog.iloc[temp_ix]

        data_sets['Xtemp'] = x_data.iloc[temp_ix].to_numpy()
        data_sets['ytemp'] = y_data.iloc[temp_ix].to_numpy()
        # group_split['temp'] = group.iloc[temp_ix]
        demographics['temp'] = demog.iloc[temp_ix]

    gss2 = StratifiedShuffleSplit(n_splits=1, train_size=.75, random_state=8)
    for train_ix, val_ix in gss2.split(x_temp, y_temp):

        split_index = [-1 if x in train_ix else 0 for x in temp_ix]
        ps = PredefinedSplit(test_fold=split_index)
        data_sets['predef_split'] = ps

        data_sets['Xtrain'] = x_temp.iloc[train_ix].to_numpy()
        data_sets['ytrain'] = y_temp.iloc[train_ix].to_numpy()
        # group_split['train'] = group_temp.iloc[train_ix]
        demographics['train'] = demog_temp.iloc[train_ix]

        data_sets['Xval'] = x_temp.iloc[val_ix].to_numpy()
        data_sets['yval'] = y_temp.iloc[val_ix].to_numpy()
        # group_split['val'] = group_temp.iloc[val_ix]
        demographics['val'] = demog_temp.iloc[val_ix]

    return data_sets, group_split, demographics

def even_out(df):
    case = df[df.label == 1]
    control = df[df.label == 0]
    new_control = control.sample(n=len(case), random_state=10)
    df = pd.concat([case, new_control], axis=0, ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    return df

def impute_knn(data, impute_type):
    """ Takes the dataset with missing data and returns an imputed dataset"""
    # imputer = SimpleImputer(strategy=impute_type)
    imputer = KNNImputer()
    data['Xtrain'] = imputer.fit_transform(data['Xtrain'])

    data['Xval'] = imputer.fit_transform(data['Xval'])
    data['Xtest'] = imputer.fit_transform(data['Xtest'])
    data['Xtemp'] = imputer.fit_transform(data['Xtemp'])

    return data

def scale_minmax(data, scale_type):
    """Scale data"""
    if scale_type == 'standard':
        scaler = StandardScaler()
    elif scale_type == 'minmax':
        scaler = MinMaxScaler()
    scaler.fit(data['Xtrain'])
    data['Xtrain'] = scaler.transform(data['Xtrain'])

    data['Xval'] = scaling.transform(data['Xval'])
    data['Xtest'] = scaling.transform(data['Xtest'])
    data['Xtemp'] = scaling.transform(data['Xtemp'])

    return data

def split_sets_tminus(data_df, test_df):

    num_case = len(data_df[data_df.label == 1])
    num_cohort = num_case * 2
    num_test_needed = int(num_cohort * 0.2)
    num_temp_needed = int(num_cohort * 0.8)

    num_case_test = int(num_test_needed / 2)  # number of cases needed in test
    num_case_temp = int(num_temp_needed / 2)  # number of controls needd in test

    after_case_mrn = test_df[test_df.label == 1].MEDICAL_RECORD_NUMBER
    after_control_mrn = test_df[test_df.label == 0].MEDICAL_RECORD_NUMBER

    case2 = data_df[data_df.label == 1]
    control2 = data_df[data_df.label == 0]

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

    return data_dict, demog_dict