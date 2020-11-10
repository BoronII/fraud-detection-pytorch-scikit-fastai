# IEEE-CIS Fraud Detection
kaggle fraud detection

## Introduction 

The dataset for this project is provided by the Vesta payment services corporation, 
as part of a competition hosted on Kaggle. The objective of the competition was to 
improve the efficacy of fraudulent transaction alerts. Models were evaluated on the 
area under the ROC curve between the predicted probability and the observed target.

## Data

The [dataset](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203) is 
anonymized and has a mix of categorical and continuous features, include 339 
features engineered by Vesta. The dataset is thoroughly explored in 01_EDA.ipynb.  

## Modelling approaches

# Baseline
As a baseline model, a gaussian naive bayes' classifier was trained.

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/GaussianNB_aucs.jpg" width="400" height="200" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/GaussianNB_cm.jpg" width="250" height="200" />

# RF

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/Important_Features_RF_aucs.jpg" width="600" height="300" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/Important_Features_RF_cm.jpg" width="400" height="300" />

# HistgradientBoostingClassifier

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/Important_Features_RF_aucs.jpg" width="600" height="300" />
<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/Important_Features_RF_aucs.jpg" width="400" height="300" />

# Neural net

## Results

AUROC vs PERCISION RECALL
- None
- Ros
- Rus
- Class weights




