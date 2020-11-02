# Module containing custom functions for IEEE Fraud Detection project

import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve



# 01_EDA (functions introduced in 01_EDA notebook)


def nans_by_col(df):
    # A table that displays missing values
    nans = pd.DataFrame(pd.Series(df.columns), columns=['column'])
    nans['NaN count'] = None
    nans['length'] = None
    nans['percent NaN'] = None
    for i, col in enumerate(df.columns):
        nans['NaN count'].iloc[i] = len(df[col]) - df[col].count()
        nans['length'].iloc[i] = len(df[col])
        nans['percent NaN'].iloc[i] = nans['NaN count'].iloc[i]/nans['length'].iloc[i]
    return nans

def summary_table(df):
    # A table that gives a high level summary of data
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    return summary

def plot_counts_fraud(df, col):
    # A double y-axis plot that displays fraud count and percent fraud
    # for categorical features by category
    total = len(df)
    total_amt = df.groupby(['isFraud'])['TransactionAmt'].sum().sum()
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    plt.figure(figsize=(16,6))    

    plot = sns.countplot(x=col, data=df, order=list(tmp[col].values))
    plot_t = plot.twinx()
    plot_t = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                           color='black', legend=False)
    plot_t.set_ylim(0,tmp['Fraud'].max()*1.1)
    plot_t.set_ylabel("%Fraud Transactions", fontsize=16)
    plot.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)
    plot.set_xlabel(f"{col} Category Names", fontsize=16)
    plot.set_ylabel("Count", fontsize=17)
    plot.set_xticklabels(plot.get_xticklabels(),rotation=45)
    sizes = []
    for p in plot.patches:
        height = p.get_height()
        sizes.append(height)
        plot.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center",fontsize=12) 
        
    plot.set_ylim(0,max(sizes)*1.15)


def plot_distribution(df, col):
    # Plots the probability distribution of values for a continuous feature
    plt.figure(figsize=(14,5))
    plot = sns.distplot(df[df['isFraud'] == 1][col], label='Fraud')
    plot = sns.distplot(df[df['isFraud'] == 0][col], label='NoFraud')
    plot.legend()
    plot.set_title(f"{col} Values Distribution by Target", fontsize=20)
    plot.set_xlabel(f"{col} Values", fontsize=18)
    plot.set_ylabel("Probability", fontsize=18)


def plot_counts_amts_and_percent_fraud(df, col):   
    # Create two plots.  
    # A plot of fraud count and percent fraud by category.  
    # A plot of total transaction amount and percent fraud by category.
    total = len(df)
    total_amt = df.groupby(['isFraud'])['TransactionAmt'].sum().sum()
   
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    plt.figure(figsize=(16,14))    
    plt.suptitle( f'{col} Distributions ', fontsize=24)

    plt.subplot(211)
    plot = sns.countplot(x=col, data=df, order=list(tmp[col].values))
    plot_t = plot.twinx()
    plot_t = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                           color='black', legend=False)
    plot_t.set_ylim(0,tmp['Fraud'].max()*1.1)
    plot_t.set_ylabel("%Fraud Transactions", fontsize=16)
    plot.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)
    plot.set_xlabel(f"{col} Category Names", fontsize=16)
    plot.set_ylabel("Count", fontsize=17)
    plot.set_xticklabels(plot.get_xticklabels(),rotation=45)
    sizes = []
    for p in plot.patches:
        height = p.get_height()
        sizes.append(height)
        plot.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center",fontsize=12) 
        
    plot.set_ylim(0,max(sizes)*1.15)
    
    perc_amt = (df.groupby(['isFraud',col])['TransactionAmt'].sum() \
                / df.groupby([col])['TransactionAmt'].sum() * 100).unstack('isFraud')
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    amt = df.groupby([col])['TransactionAmt'].sum().reset_index()
    perc_amt = perc_amt.fillna(0)
    
    plt.subplot(212)
    plot1 = sns.barplot(x=col, y='TransactionAmt', 
                       data=amt, 
                       order=list(tmp[col].values))
    plot1_t = plot1.twinx()
    plot1_t = sns.pointplot(x=col, y='Fraud', data=perc_amt, 
                        order=list(tmp[col].values),
                       color='black', legend=False, )
    plot1_t.set_ylim(0,perc_amt['Fraud'].max()*1.1)
    plot1_t.set_ylabel("%Fraud Total Amount", fontsize=16)
    plot1.set_xticklabels(plot1.get_xticklabels(),rotation=45)
    plot1.set_title(f"{col} by Transactions Total + %of total and %Fraud Transactions", fontsize=20)
    plot1.set_xlabel(f"{col} Category Names", fontsize=16)
    plot1.set_ylabel("Transaction Total Amount(U$)", fontsize=16)
    
    for p in plot1.patches:
        height = p.get_height()
        plot1.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total_amt*100),
                ha="center",fontsize=12) 
        
    plt.subplots_adjust(hspace=.4, top = 0.9)
    plt.show()


def plot_dist_ratio(df, col, lim=2000):
    total = len(df)
    total_amt = df.groupby(['isFraud'])['TransactionAmt'].sum().sum()
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.figure(figsize=(20,5))
    plt.suptitle(f'{col} Distributions ', fontsize=22)

    plt.subplot(121)
    plot = sns.countplot(x=col, data=df, order=list(tmp[col].values))
    plot.set_title(f"{col} Distribution\nCount and %Fraud by each category", fontsize=18)
    plot.set_ylim(0,400000)
    plot_t = plot.twinx()
    plot_t = sns.pointplot(x=col, y='Fraud', data=tmp, order=list(tmp[col].values),
                       color='black', legend=False, )
    plot_t.set_ylim(0,20)
    plot_t.set_ylabel("% of Fraud Transactions", fontsize=16)
    plot.set_xlabel(f"{col} Category Names", fontsize=16)
    plot.set_ylabel("Count", fontsize=17)
    for p in plot_t.patches:
        height = p.get_height()
        plot_t.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center",fontsize=14) 
        
    perc_amt = (df.groupby(['isFraud',col])['TransactionAmt'].sum() / total_amt * 100).unstack('isFraud')
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    plt.subplot(122)
    plot1 = sns.boxplot(x=col, y='TransactionAmt', hue='isFraud', 
                     data=df[df['TransactionAmt'] <= lim], order=list(tmp[col].values))
    plot1_t = plot1.twinx()
    plot1_t = sns.pointplot(x=col, y='Fraud', data=perc_amt, order=list(tmp[col].values),
                       color='black', legend=False, )
    plot1_t.set_ylim(0,5)
    plot1_t.set_ylabel("%Fraud Total Amount", fontsize=16)
    plot1.set_title(f"{col} by Transactions dist", fontsize=18)
    plot1.set_xlabel(f"{col} Category Names", fontsize=16)
    plot1.set_ylabel("Transaction Amount(U$)", fontsize=16)
        
    plt.subplots_adjust(hspace=.4, wspace = 0.35, top = 0.80)

#02_RF 

def fit(m, xs, y, valid_xs, valid_y):
    # Fit model
    m.fit(xs, y)
    # Make predictions
    prob_train = m.predict_proba(xs)[:,1]
    pred_train = m.predict(xs)
    prob_valid = m.predict_proba(valid_xs)[:,1]
    pred_valid = m.predict(valid_xs)
    # Calculate AROC for training and validation
    Train_AROC = roc_auc_score(y, prob_train )
    Valid_AROC = roc_auc_score(valid_y, prob_valid)
    #  Print AROC for training and validation
    print(f'Train_AROC: {Train_AROC}')
    print(f'Valid_AROC: {Valid_AROC}')
    # Return predictions
    return prob_train, pred_train, prob_valid, pred_valid

def metrics(y_true, y_pred, y_prob, acc=False):
    # calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    P, R, F1, _  = precision_recall_fscore_support(y_true, y_pred, beta=1.0)
    _, _, F2, _  = precision_recall_fscore_support(y_true, y_pred, beta=2.0)
    _, _, F5, _  = precision_recall_fscore_support(y_true, y_pred, beta=5.0)
    _, _, F10, _ = precision_recall_fscore_support(y_true, y_pred, beta=10.0)
    AROC         = roc_auc_score(y_true, y_prob)
    APRC         = auc(recall, precision)
    if acc==True:
        cm = confusion_matrix(y_true, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        ACC = (TP+TN)/(TP+FP+FN+TN)
    print(f'AROC: {AROC}')  
    print(f'APRC: {APRC}')
    print(f'Percision: {P[1]}')
    print(f'Recall: {R[1]}')
    if acc==True:
        print(f'ACC: {ACC[1]}')
    print(f'F1: {F1[1]}')
    print(f'F2: {F2[1]}')
    print(f'F5: {F5[1]}')
    print(f'F10: {F10[1]}')
    return P[1], R[1], F1[1], F2[1], F5[1], F10[1] 

def plot_confusion_matrix(y_true, y_pred):
    # calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = ['legit', 'Fraud']
    fig = plt.figure()
    #ax = fig.add_subplot(111)
    df_cm = pd.DataFrame(cm, range(len(cm)), range(len(cm)))
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='.0f') # font size
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    return fig

def plot_roc_prc(y_true, y_prob, fig_size=(11,5), label=None):
    fig, ax = plt.subplots(1,2, figsize=fig_size)    

    # precision-recall curve
    # calculate the no skill line as the proportion of the positive class
    no_skill = len(y_true[y_true==1]) / len(y_true)
    # calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # plot the no skill precision-recall curve
    ax[0].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # plot the precision-recall curve
    if label!=None:
        ax[0].plot(recall, precision, marker='.', label=label)
    else:
        ax[0].plot(recall, precision, marker='.')
    # axis labels
    ax[0].set(xlabel='Precision', ylabel='Recall')
    # show the legend
    ax[0].legend()

    # roc curve
    # plot no skill roc curve
    ax[1].plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # calculate roc curve 
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    # plot roc curve
    if label!=None:
        ax[1].plot(fpr, tpr, marker='.', label=label)
    else:
        ax[1].plot(fpr, tpr, marker='.')
    # axis labels
    ax[1].set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    # show the legend
    ax[1].legend()
    return fig




