import pandas as pd

from skclassifier           import SKClassifier
from fastai.tabular.all     import TabularPandas, Categorify, FillMissing
from sklearn.ensemble       import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

class RFClassifier(SKClassifier):
      
    def __init__(self, model, df, cat, cont, dep_var, procs):
        super().__init__(model, df, cat, cont, dep_var, procs) 
    
    # Calculate feature importance
    # Sort features by importance
    def feature_importances(self, df,  thresh, to_drop=None):
        fi = pd.DataFrame({'cols': df.columns,'imp': self.model.feature_importances_}) \
                          .sort_values('imp', ascending=False)
        # Keep only features with importance exceeding the threshold
        to_keep           = fi[fi.imp>thresh].cols
        self.xs_imp       = self.xs[to_keep]
        self.valid_xs_imp = self.valid_xs[to_keep]
        
        # Optionaly, drop a list of features
        if to_drop != None: 
            self.xs_imp       = self.xs_imp.drop(to_drop, axis=1)
            self.valid_xs_imp = self.valid_xs_imp.drop(to_drop, axis=1)
        
        # Print the number of features 
        print(f'n_features: {len(to_keep)}')
        return self.xs_imp, self.valid_xs_imp

     
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
    
    model = RandomForestClassifier(n_jobs=-1, max_samples=2/3,
                           oob_score=True, max_features='sqrt',
                           n_estimators=1000, criterion='entropy',
                           max_leaf_nodes=750, min_samples_split=30, min_samples_leaf=5)   

    clf = RFClassifier(model, TRAIN, CAT, CONT, DEP_VAR, PROCS)
    
    clf.process_df()

    xs, y             = clf.xs,       clf.y
    valid_xs, valid_y = clf.valid_xs, clf.valid_y
    
    clf.fit(xs, y)
    
    # This feature was found to be redundant with the TransactionDT feature
    to_drop = ['TransactionID']    

    # Get training and validation dataframes with important features
    # The threshold was determined experimentally (see notebooks/02...)
    xs_imp, valid_xs_imp = clf.feature_importances(xs, thresh=0.003, to_drop=to_drop)

    ros = RandomOverSampler(random_state=42)
    
    # Apply over sampling to training dataframes
    xs_imp, y = ros.fit_resample(xs_imp, y)
   
    assert y.sum()==len(y)/2
   
    clf.fit(xs_imp, y)
    
    train_preds = clf.predict_proba(xs_imp)
    valid_preds = clf.predict_proba(valid_xs_imp)
    
    train_auroc = clf.auroc(y, train_preds)
    print(f'Train_AUROC: {train_auroc}')
    
    valid_auroc = clf.auroc(valid_y, valid_preds)
    print(f'Valid_AUROC: {valid_auroc}')

    clf.save('models/RF_model.pkl')
    
    
   
