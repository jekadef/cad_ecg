from sklearn.model_selection import train_test_split, PredefinedSplit, GroupShuffleSplit, StratifiedShuffleSplit


# def get_sets(x_data, y_data):
#     """Takes the dataset and splits into training, validation, test and predefinted splits"""
#     data_sets = {}
#     Xtemp, Xtest, ytemp, ytest = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data, random_state=8)
#     Xtrain, Xval, ytrain, yval = train_test_split(Xtemp, ytemp, test_size=0.25, random_state=8)
#     split_index = [-1 if x in Xtrain.index else 0 for x in Xtemp.index]
#     ps = PredefinedSplit(test_fold=split_index)
#     data_sets['Xtrain'] = Xtrain.to_numpy()
#     data_sets['ytrain'] = ytrain.to_numpy()
#     data_sets['Xval'] = Xval.to_numpy()
#     data_sets['yval'] = yval.to_numpy()
#     data_sets['Xtest'] = Xtest.to_numpy()
#     data_sets['ytest'] = ytest.to_numpy()
#     data_sets['splits'] = ps
#     data_sets['Xtemp'] = Xtemp
#     data_sets['ytemp'] = ytemp
#     train_index = Xtrain.index
#     val_index = Xval.index
#     test_index = Xtest.index
#     return data_sets, train_index, val_index, test_index

def get_sets(data_df):
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

# def get_sets(data_df):
#     """Takes the dataset and splits into training, validation, test and predefinted splits"""
#     data_sets = {}
#     group_split = {}
#     demographics = {}
#     gss1 = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=8)
#     x_data = data_df.loc[:, ['VentricularRate', 'AtrialRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
#                              'PAxis', 'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset']]
#     y_data = data_df.loc[:, 'label']
#     demog = data_df.loc[:, ['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate', 'MEDICAL_RECORD_NUMBER',
#                             'PERSON_KEY', 'GENDER', 'RACE', 'PATIENT_ETHNIC_GROUP', 'DATE_OF_BIRTH', 'CALENDAR_DATE',
#                             'GROUP_RACE_ETHNICITY', 'META_GROUP', 'full_path', 'label', 'time_delta', 'years_icd_ecg',
#                             'age_binned', 'weights', 'CONTEXT_DIAGNOSIS_CODE']]
#     group = data_df.loc[:, 'patientid']
#     for temp_ix, test_ix in gss1.split(x_data, y_data, group):
#
#         data_sets['Xtest'] = x_data.iloc[test_ix].to_numpy()
#         data_sets['ytest'] = y_data.iloc[test_ix].to_numpy()
#         group_split['test'] = group.iloc[test_ix]
#         demographics['test'] = demog.iloc[test_ix]
#
#         x_temp = x_data.iloc[temp_ix]
#         y_temp = y_data.iloc[temp_ix]
#         group_temp = group.iloc[temp_ix]
#         demog_temp = demog.iloc[temp_ix]
#
#         data_sets['Xtemp'] = x_data.iloc[temp_ix].to_numpy()
#         data_sets['ytemp'] = y_data.iloc[temp_ix].to_numpy()
#         group_split['temp'] = group.iloc[temp_ix]
#         demographics['temp'] = demog.iloc[temp_ix]
#
#     gss2 = GroupShuffleSplit(n_splits=1, train_size=.75, random_state=8)
#     for train_ix, val_ix in gss2.split(x_temp, y_temp, group_temp):
#
#         split_index = [-1 if x in train_ix else 0 for x in temp_ix]
#         ps = PredefinedSplit(test_fold=split_index)
#         data_sets['predef_split'] = ps
#
#         data_sets['Xtrain'] = x_temp.iloc[train_ix].to_numpy()
#         data_sets['ytrain'] = y_temp.iloc[train_ix].to_numpy()
#         group_split['train'] = group_temp.iloc[train_ix]
#         demographics['train'] = demog_temp.iloc[train_ix]
#
#         data_sets['Xval'] = x_temp.iloc[val_ix].to_numpy()
#         data_sets['yval'] = y_temp.iloc[val_ix].to_numpy()
#         group_split['val'] = group_temp.iloc[val_ix]
#         demographics['val'] = demog.iloc[val_ix]
#
#     return data_sets, group_split, demographics

# def get_sets(x_data, y_data, group):
#     """Takes the dataset and splits into training, validation, test and predefinted splits"""
#     data_sets = {}
#     group_split = {}
#
#     gss1 = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=8)
#
#     for temp_ix, test_ix in gss1.split(x_data, y_data, group):
#         x_data.reset_index(inplace=True, drop=True)
#         y_data.reset_index(inplace=True, drop=True)
#         group.reset_index(inplace=True, drop=True)
#
#         data_sets['Xtest'] = x_data.iloc[test_ix].to_numpy()
#         data_sets['ytest'] = y_data.iloc[test_ix].to_numpy()
#         group_split['test'] = group[test_ix]
#
#         x_temp = x_data.iloc[temp_ix]
#         y_temp = y_data.iloc[temp_ix]
#         group_temp = group.iloc[temp_ix]
#
#     gss2 = GroupShuffleSplit(n_splits=1, train_size=.75, random_state=8)
#
#     for train_ix, val_ix in gss2.split(x_temp, y_temp, group_temp):
#         x_temp.reset_index(inplace=True, drop=True)
#         y_temp.reset_index(inplace=True, drop=True)
#         group_temp.reset_index(inplace=True, drop=True)
#
#         data_sets['Xtrain'] = x_temp.iloc[train_ix].to_numpy()
#         data_sets['ytrain'] = y_temp.iloc[train_ix].to_numpy()
#         group_split['train'] = group_temp[train_ix]
#
#         data_sets['Xval'] = x_temp.iloc[val_ix].to_numpy()
#         data_sets['yval'] = y_temp.iloc[val_ix].to_numpy()
#         group_split['val'] = group_temp[val_ix]
#
#     return data_sets, group_split
