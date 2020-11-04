import math
import numpy  as np
import pandas as pd
import pickle

from fastai.tabular.all  import TabularPandas, Categorify, FillMissing  
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
        
        # Get training and validation data
        self.xs, self.y             = to.train.xs, to.train.y
        self.valid_xs, self.valid_y = to.valid.xs, to.valid.y
        return to
           
        
    def fit(self, xs, y):
        # Train classifier
        self.model.fit(xs, y)
    
    def predict_proba(self, xs):
        # Get predictions
        self.prob = self.model.predict_proba(xs)[:,1]
        return self.prob
        
    def auroc(self, targ, pred):
        # Calculate auroc
        auroc = roc_auc_score(targ, pred)
        return auroc 
        
    def save(self, path): 
        path = Path(path)
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)
