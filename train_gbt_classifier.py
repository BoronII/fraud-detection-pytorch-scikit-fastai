import pandas as pd

from SKClassifier         import SKClassifier
from fastai.tabular.all   import TabularPandas, Categorify, FillMissing
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble     import HistGradientBoostingClassifier
from pathlib              import Path 
     
if __name__=='__main__':

    PATH = Path('data')
    
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
    
    model = HistGradientBoostingClassifier(loss='binary_crossentropy', verbose=1, 
                                           l2_regularization=2.4, learning_rate=0.03,          
                                           max_depth=25, max_iter=1000, 
                                           max_leaf_nodes=44, min_samples_leaf=8,  
                                           scoring='roc_auc', tol=1e-8)   

    clf = SKClassifier(model, TRAIN, CAT, CONT, DEP_VAR, PROCS)
    
    clf.process_df()
   
    xs, y             = (PATH/'xs_imp.pkl').load(),       clf.y
    valid_xs, valid_y = (PATH/'valid_xs_imp.pkl').load(), clf.valid_y
    
    clf.fit(xs, y)   
        
    train_preds = clf.predict_proba(xs)
    valid_preds = clf.predict_proba(valid_xs)
    
    train_auroc = clf.auroc(y, train_preds)
    print(f'Train_AUROC: {train_auroc}')
    
    valid_auroc = clf.auroc(valid_y, valid_preds)
    print(f'Valid_AUROC: {valid_auroc}')

    clf.save('models/GBT_model.pkl')
