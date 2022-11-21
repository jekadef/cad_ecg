import gzip
import pickle as pkl
import torch
from sklearn.model_selection import train_test_split
import pandas as pd

# working directory
data_dir = '/hpc/users/defrej02/projects/cardio_phenotyping/data/data'
cohort_file = data_dir + 'case_control_cad_20210913.pkl.gz'

# load file with filenames of ECGs and their labels (previously created in "cohort_selection" directory)
with gzip.GzipFile(cohort_file, 'rb') as f:
    cohort = pkl.load(f)
cohort['full_path'] = cohort.path + '/' + cohort.filename

# specify filename and label
x_data = cohort.iloc[:, 8]
y_data = cohort.iloc[:, 7]

# split data into training validation and testing sets
Xtemp, Xtest, ytemp, ytest = train_test_split(x_data, y_data, test_size=0.2, stratify=y_data, random_state=8)
Xtrain, Xval, ytrain, yval = train_test_split(Xtemp, ytemp, test_size=0.25, stratify=ytemp, random_state=8)

train_ds = pd.concat([Xtrain, ytrain], axis=1)
val_ds = pd.concat([Xval, yval], axis=1)
test_ds = pd.concat([Xtest, ytest], axis=1)

# save datasets
with open(data_dir + 'training_ecg_dataset.pkl', 'wb') as f:
    pkl.dump(train_ds, f)

with open(data_dir + 'validation_ecg_dataset.pkl', 'wb') as f:
    pkl.dump(val_ds, f)

with open(data_dir + 'testing_ecg_dataset.pkl', 'wb') as f:
    pkl.dump(test_ds, f)
