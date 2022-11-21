# Phenotyping Coronary Artery Disease from Electrocardiograms
#### This project aims to detect CAD from ECG data using deep learning. We hypothesize that a convolutional neural network architecture can detect ...

Python version 3.8

Packages required: requirements.txt

#Module descriptions


### 1a. run_cardio_dx_query.sh & cardio_dx_query.sql
- identify patients with CAD from MSDW
- On HPIMS server
- mysql db
- Resulting in file "cardio_dx_msdw_20200407.txt"
- scp the file to MSCIC VM


### 1b. get_mrn_filenames_ge_ecg.sql
- get files names for all XMLs 
- On MSCIC server
- postgresql db
- Resulting in file "ecg_mrn_filenames.csv"
  
### 2. get_cohort_filenames.py
- join cardio_dx_msdw.txt with ecg_mrn_filenames.csv
- subset with cardio disease of interest
- Resulting in file "case_cad_YYYYMMDD.pkl.gz" & "control_cad_YYYYMMDD.pkl.gz"

### 3. get_filtered_cohort.py
- get measurement data from all cases and equal number of controls
- Resulting in file "case_control_cad_YYYYMMDD.pkl.gz"

### 4a. get_measurement_dataset.py
- get measurement data from cohort

### 4b. get_waveform_dataloader.py
- for the CNN

### 5. main_hyperparameters.py
- find best model parameters for baseline models
- resulting in 

### 6. main_performance.py
- get performance of best baseline models
- resulting in

#### utils.py
- filenames

#### preprocess.py
- functions for filtering, imputation, scaling data

#### split.py
- create the train validation testing sets

#### search.py
- find best hyperparameters by grid search

#### evaluate.py
- train models with best hyperparameters and predict

#### metrics.py
- get performance 

#### visualization.py
- plots


