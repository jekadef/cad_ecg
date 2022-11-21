import xmltodict
import numpy as np
import pickle as pkl
import base64
import pandas as pd
import utils
import gzip


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
    # filtered_cohort_files = filtered_cohort.loc[:, ['full_path', 'label']]

    # return list_measurements, filtered_cohort, filtered_cohort_files
    return list_measurements, filtered_cohort


def save_cohort_dataset(measure_list, cohort_filtered_df, time_bin, group):
    measurement_df = pd.DataFrame.from_records(measure_list)
    ecg_measurement = pd.concat([cohort_filtered_df.reset_index(drop=True), measurement_df], axis=1)

    timebin = str(time_bin).replace('(', '_').replace(', ', '_').replace(']', '_')
    cohort_measurement_fn = utils.cohort_selection_workingdir + group + "_measurement_dataset_timedelta" + timebin + "20220824.pkl.gz"

    with gzip.open(cohort_measurement_fn, 'wb') as f:
        pkl.dump((ecg_measurement.columns, ecg_measurement.to_numpy()), f)

    return ecg_measurement


def split_measurement_by_timeframe(df, window):
    window_start = -window
    window_end = window
    df['timeframe_binned'] = 'time'
    df.loc[(df.time_delta.dt.days < window_start), 'timeframe_binned'] = 't_before'
    df.loc[(df.time_delta.dt.days >= window_start) & (df.time_delta.dt.days <= window_end), 'timeframe_binned'] = 't_zero'
    df.loc[(df.time_delta.dt.days > window_end), 'timeframe_binned'] = 't_after'

    # only save those in before and after df after accounting for the specific window ***TO DO****
    measurement_dict = {'t_before': df.loc[df.timeframe_binned == 't_before'],
                        't_zero': df.loc[df.timeframe_binned == 't_zero'],
                        't_after': df.loc[df.timeframe_binned == 't_after']}

    return measurement_dict

def split_measurement_before_after(df, start, end):
    window_start = start
    window_end = end
    df['timeframe_binned'] = 'time'
    df.loc[(df.time_delta.dt.days < window_start), 'timeframe_binned'] = 't_before'
    df.loc[(df.time_delta.dt.days >= window_start) & (df.time_delta.dt.days <= window_end), 'timeframe_binned'] = 't_zero'
    df.loc[(df.time_delta.dt.days > window_end), 'timeframe_binned'] = 't_after'

    # only save those in before and after df after accounting for the specific window ***TO DO****
    measurement_dict = {'t_before': df.loc[df.timeframe_binned == 't_before'],
                        't_zero': df.loc[df.timeframe_binned == 't_zero'],
                        't_after': df.loc[df.timeframe_binned == 't_after']}

    return measurement_dict

