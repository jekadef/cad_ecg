# Real-world Challenges in Leveraging Electrocardiograms for Coronary Artery Disease Classification
#### This project aims to detect CAD from ECG data using deep learning and investigate the challenges that comes with this task.

This work was presented at the Time-series for Health workshop at NeurIPS 2022

Python version 3.8

Packages required: requirements.txt

#Module descriptions

## Cohort Selection
- functions: utils.py, mapping.py, process.py

### 01 Data Extraction from Database
- Execute: run_cardio_dx_query.sh 
- SQL query: **cardio_dx_query.sql**
- Goal is to identify patients with CAD from MSDW
- Using mysql db HPIMS server
- Resulting in file "cardio_dx_msdw_20200407.txt"
- scp the file to MSCIC VM

### 02 Identification of Cases 
- Code: **get_cases_demographics_20221102.py**
- Execute files in directory: 02_get_cases_demographics
- Output:

### 03 Identification of Controls
- Code: **get_control_demographics_20221102.py**
- Execute files in directory: 03_get_control_demographics
- Output:

### 04 Create t-plus Dataset
- Code: get_input_tplus_20221102.py
- Execute files in directory: 04_get_tplus_input
- Output:

### 05 Create t-minus Datasets
- Code: get_input_tminus_20221102.py
- Execute files in directory: 05_get_tminus_input 
- Output:

## Baseline Models
- functions: utils.py, evaluate.py, preprocess.py, search.py, split.py, visualization.py, 

### 06 Hyperparameter Search
- Code: byperparameter_search_20221102.py
- Execute: 06_search_hyperparameters
- Output:

### 07 Train Baseline Models 
- Code: baseline_training.py
- Execute: 07_train_baselines
- Output:

### 08 Evaluate Baseline Models
- Code: baseline_evaluate.py
- Execute: 08_evaluate_baselines
- Output:

## Neural Network Models
- functions: dataset.py, model.py, preprocess.py, utill.py

### 09 Train CNNs
- Code: train.py
- Execute: 09_train_cnn
- Output:

### 10 Evaluate CNNs
- Code: evlauate_pretrained_model.py
- Execute: 10_evaluate_cnn
- Output:
