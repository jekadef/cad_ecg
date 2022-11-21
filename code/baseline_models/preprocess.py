from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer


def load_process_data(df, drop_vars):
    demographics = df[['path', 'filename', 'patientid', 'age', 'gender', 'acquisitiondate', 'MEDICAL_RECORD_NUMBER',
                    'PERSON_KEY', 'GENDER', 'RACE', 'PATIENT_ETHNIC_GROUP', 'DATE_OF_BIRTH', 'CALENDAR_DATE',
                    'GROUP_RACE_ETHNICITY', 'META_GROUP', 'full_path', 'time_delta', 'years_icd_ecg',
                    'age_binned', 'weights', 'CONTEXT_DIAGNOSIS_CODE']]
    # variables to include in the model
    data = df[['label', 'VentricularRate', 'AtrialRate', 'PRInterval', 'QRSDuration', 'QTInterval', 'QTCorrected',
               'PAxis', 'RAxis', 'TAxis', 'QRSCount', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset',
               'ECGSampleBase', 'ECGSampleExponent', 'QTcFrederica', 'SystolicBP', 'DiastolicBP']]
    data = data.astype(float)
    data['label'] = data['label'].astype(int)
    # drop variables that were chosen in the missingness analysis
    data = data.drop(drop_vars, axis='columns')
    return demographics, data


def quantile_filter(x_data, low, high):
    low_quantile = x_data.quantile(low, axis='rows')
    high_quantile = x_data.quantile(high, axis='rows')

    for col in x_data.columns:
        x_data = x_data.loc[((x_data[col] >= low_quantile[col]) & (x_data[col] <= high_quantile[col])) | x_data[col].isna()]

    return x_data


def get_imputed(x_data, impute_type):
    """ Takes the dataset with missing data and returns an imputed dataset"""
    # imputer = SimpleImputer(strategy=impute_type)
    imputer = KNNImputer()
    imputed_data = imputer.fit_transform(x_data)
    return imputer, imputed_data


def get_scaling(x_data, scale_type):
    """Scale data"""
    if scale_type == 'standard':
        scaler = StandardScaler()
    elif scale_type == 'minmax':
        scaler = MinMaxScaler()
    scaler.fit(x_data)
    scaled_data = scaler.transform(x_data)
    return scaler, scaled_data
