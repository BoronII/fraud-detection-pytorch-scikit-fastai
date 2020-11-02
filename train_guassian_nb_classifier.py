import pandas as pd
import numpy  as np
import pickle 
import math

from fastai.tabular.all import TabularPandas
from fastai.tabular.all import Categorify
from fastai.tabular.all import FillMissing
  
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

train = pd.read_csv('data/train.csv', index_col=[0], low_memory=False)

# Categorical features
cat = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
       'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
       'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'DeviceType', 'DeviceInfo', 'id_12', 
       'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 
       'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
       'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36',
       'id_37', 'id_38'] 

# Continuous features
cont = ['TransactionID', 'TransactionAmt', 'TransactionDT',
        'dist1', 'dist2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 
        'C10', 'C11','C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 
        'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'id_01', 'id_02', 
        'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 
        'id_11']
# Target feature
dep_var = 'isFraud'

# Calculate time delta on which to split data into training and validation
cut_off = math.floor(train['TransactionDT'].min() + 
                    (train['TransactionDT'].max() - train['TransactionDT'].min())*0.8)

# The rows with TransactionID <= cut_off comprise the first 80% of the data (approximately)
cond = train['TransactionDT'] <= cut_off
train_idx = np.where( cond)[0]
valid_idx = np.where(~cond)[0]
splits = (list(train_idx),list(valid_idx))

# Data processing procedures
# Categorify replaces categorical columns with numerical categorical columns. 
# FillMissing replaces missing values with the median of the column and 
# creates a new Boolean column that records whether data was missing.
procs = [Categorify, FillMissing]

# Tabular object
to = TabularPandas(train, procs, cat, cont, y_names=dep_var, splits=splits,
                   do_setup=True, reduce_memory=True)

# Get training and validation data
xs, y             = to.train.xs, to.train.y
valid_xs, valid_y = to.valid.xs, to.valid.y

# Train Gaussian Naive Bayes' classifier

m = GaussianNB()

m.fit(xs, y)

# Get predictions
prob_train = m.predict_proba(xs)[:,1]
prob_valid = m.predict_proba(valid_xs)[:,1]

# Calculate AROC
Train_AROC = roc_auc_score(y,       prob_train)
Valid_AROC = roc_auc_score(valid_y, prob_valid)

# Print AROC 
print(f'Train_AROC: {Train_AROC}')
print(f'Valid_AROC: {Valid_AROC}')

with open('models/GNB_model.pkl', 'wb') as file:
    pickle.dump(m, file)

