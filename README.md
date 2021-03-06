# IEEE-CIS Fraud Detection
Kaggle fraud detection

## Introduction 

The dataset for this project is provided by the Vesta payment services corporation, 
as part of a competition hosted on Kaggle. The objective of the competition was to 
improve the efficacy of fraudulent transaction alerts. Models were evaluated on the 
area under the ROC curve for the predicted probability and the observed target.

## Data

The [dataset](https://www.kaggle.com/c/ieee-fraud-detection/discussion/101203) is 
anonymized and has a mix of categorical and continuous features, include 339 
features engineered by Vesta. The dataset is thoroughly explored in 01_EDA.ipynb. 
Two notable features of the dataset are the class imbalance (approximately 30:1) 
and, the distribution of transaction time deltas. All of the transactions from the 
test set are in the future with respect to the training set. This prompted splitting 
off of the most recent 20% of the training data for validation. 

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/ClassDistribution.png" width="400" height="200" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/TransactionDT.png" width="400" height="200" />

## Modelling approaches

Details regarding each of the modelling approaches can be found in the notebooks. 
Where applicable, hyperparameters were tuned and the best model with respect to roc score was selected.
The training scripts are contained in the main directory.

# Baseline

As a baseline model, a gaussian naive bayes' classifier was trained.

This model achieved an roc score of 0.71. 
The precision and recall were both under 12%.

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/GaussianNB_aucs.jpg" width="500" height="250" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/GaussianNB_cm.jpg" width="300" height="250" />

# RF

With tabular data, it is always a good idea to try some tree based models. 
Another advantage of random forests is that there are several packages for 
sklearn that aid in their interpretation. For instance, feature importance analysis 
using the random forest model was used to reduce the number of features under 
consideration, while maintaining the roc score. 

This model achieved an roc score of 0.88.
The percision was nearly 40% and the recall was approximately 46%.

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/Important_Features_RF_aucs.jpg" width="500" height="250" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/Important_Features_RF_cm.jpg" width="300" height="250" />

Random over-sampling and random under-sampling resulted in models with approximately the 
same roc score as the model that was finally chosen (which does not attempt to correct the class imbalance).
**Random over-sampling** led to a model with 32% precision and 45% recall.

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/ROS_Imp_RF_auc.jpg" width="500" height="250" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/ROS_Imp_RF_cm.jpg" width="300" height="250" />

While **random under-sampling** led to a model with 13% percision and 75% recall.

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/RUS_Imp_RF_aucs.jpg" width="500" height="250" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/RUS_Imp_RF_cm.jpg" width="300" height="250" />


# HistGradientBoostingClassifier
HistGradientBoostingClassifier is an implementation (by the sklearn team) of microsoft's LightGBM. 

This model achieved an roc score of 0.88.

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/Imp_GBT_aucs.jpg" width="500" height="250" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/Imp_GBT_vm.jpg" width="300" height="250" />

**Random over-sampling** led to a model with an roc score of 0.87 , 47% percision and, 40% recall.

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/ROS_Imp_GBT_aucs.jpg" width="500" height="250" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/ROS_Imp_GBT_cm.jpg" width="300" height="250" />

**Random under-sampling** led to a model with roc score of 0.86, 10% percision and, 77% recall.

<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/RUS_Imp_GBT_aucs.jpg" width="500" height="250" /> <img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/RUS_Imp_GBT_cm.jpg" width="300" height="250" />


# Neural net
When there are categorical variables with large cardinality (many levels). 
Neural networks sometimes out perform tree based models. 

This model achieved an roc score of 0.82.


<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/classweighted_nn_metrics.png" width="800" height="200" /> 
<img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/NN_cm.jpg" width="300" height="250" />

The figure below displays the confusion matrix and auroc for the neural net in the 4 imbalance correction conditions.

When compared to no imbalance correction, all 3 imbalance correction methods come with an increase in recall at the cost of percision.

<div style="text-align:center"><img src="https://github.com/BoronII/fraud-detection-pytorch-scikit-fastai/blob/master/figures/NNs.jpg" width="600" height="500" /></div>



# Conclusions

The two best performing models with respect to roc score turned out to be the random forest and the Gradient boosted trees model, 
which each achieved a roc score of approximately 0.88. The random forest model was indispensable because of the insights that it
provided about the dataset. The gradient boosted trees model was created using a sklearn implementation (HistGradientBoostingClassifier) of Microsofts LightGBM. LightGBM uses a couple of approximations that make it computationally very efficient when applied to large datasets with many features.
In that setting, the HistGradientBoostingClassifier may be a good choice.

The imbalance correction methods investigated here (over/under-sampling and class weighting) did not have a great impact on the roc scores of the models. 
The main impact of the imbalance correction methods was on the balance between precision and recall. 



