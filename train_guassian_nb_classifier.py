import math
import numpy  as np
import pandas as pd
import pickle 

from fastai.tabular.all  import TabularPandas, Categorify, FillMissing
  
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics     import roc_auc_score
from pathlib             import Path

class SKClassifier: 

    def __init__(self, model, df, cat, cont, dep_var, procs): 
        self.cat     = cat
        self.cont    = cont
        self.dep_var = dep_var
        self.procs   = procs
        self.df      = df
        self.model   = model
        
    def process_df(self):
        # Calculate time delta on which to split data into training and validation
        cut_off = math.floor(self.df['TransactionDT'].min() + 
                            (self.df['TransactionDT'].max() - self.df['TransactionDT'].min())*0.8)
        
        # The rows with TransactionID <= cut_off comprise the first 80% of the data (approximately)
        cond = self.df['TransactionDT'] <= cut_off
        train_idx = np.where( cond)[0]
        valid_idx = np.where(~cond)[0]
        splits = (list(train_idx),list(valid_idx))
        
        # Tabular object
        to = TabularPandas(self.df, self.procs, self.cat, self.cont, 
                           y_names=self.dep_var, splits=splits, do_setup=True, 
                           reduce_memory=True)
        return to
           
        
    def fit(self, to):
        # Get training and validation data
        self.xs, self.y             = to.train.xs, to.train.y
        self.valid_xs, self.valid_y = to.valid.xs, to.valid.y
    
        # Train Gaussian Naive Bayes' classifier
        self.model.fit(self.xs, self.y)
    
    def predict_proba(self):
        # Get predictions
        self.prob_train = self.model.predict_proba(self.xs)[:,1]
        self.prob_valid = self.model.predict_proba(self.valid_xs)[:,1]
        return self.prob_train, self.prob_valid
        
    def auroc(self):
        # Calculate auroc
        train_auroc = roc_auc_score(self.y,       self.prob_train)
        valid_auroc = roc_auc_score(self.valid_y, self.prob_valid)
        
        # Print auroc 
        print(f'Train_AROC: {train_auroc}')
        print(f'Valid_AROC: {valid_auroc}')
        
    def save(self, path): 
        path = Path(path)
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)


if __name__=='__main__':
    
    TRAIN = pd.read_csv('data/train_s.csv', index_col=[0], low_memory=False)
    
    # Drop columns with V*** features
    TRAIN = TRAIN.drop(list(TRAIN.filter(regex = 'V')), axis = 1)

    # Categorical features
    CAT = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
           'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
           'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'DeviceType', 'DeviceInfo', 'id_12', 
           'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 
           'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 
           'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36',
           'id_37', 'id_38'] 
    
    # Continuous features
    CONT = ['TransactionID', 'TransactionAmt', 'TransactionDT',
            'dist1', 'dist2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 
            'C10', 'C11','C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 
            'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'id_01', 'id_02', 
            'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 
            'id_11']
    
    # Target feature
    DEP_VAR = 'isFraud'

    assert len(CAT)+len(CONT)==(len(TRAIN.columns)-1)
    
    # Data processing procedures
    # Categorify replaces categorical columns with numerical categorical columns. 
    # FillMissing replaces missing values with the median of the column and 
    # creates a new Boolean column that records whether data was missing.
    PROCS = [Categorify, FillMissing]
    
    model = GaussianNB()   

    clf = SKClassifier(model, TRAIN, CAT, CONT, DEP_VAR, PROCS)
    
    TO = clf.process_df()
    
    clf.fit(TO)
    clf.predict_proba()
    clf.auroc()
    clf.save('models/GNB_model.pkl')
    
    
    
    
    
        