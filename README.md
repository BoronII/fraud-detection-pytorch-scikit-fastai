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

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/GaussianNB_aucs.jpg" width="500" height="250" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/GaussianNB_cm.jpg" width="300" height="250" />

# RF
With tabular data it is always a good idea to try some tree based models.

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/Important_Features_RF_aucs.jpg" width="500" height="250" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/Important_Features_RF_cm.jpg" width="300" height="250" />

# HistGradientBoostingClassifier
HistGradientBoostingClassifier is an implementation (by the sklearn team) of microsoft's LightGBM. 

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/Imp_GBT_aucs.jpg" width="1200" height="250" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/Imp_GBT_vm.jpg" width="300" height="250" />

# Neural net
When there are categorical variables with large cardinality (many levels). Neural networks sometimes out perform tree based models. 

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/classweighted_nn_metrics.png" width="500" height="250" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/classweighted_nn_cm.png" width="300" height="250" />
## Results

AUROC vs PERCISION RECALL
- None
- Ros
- Rus
- Class weights




