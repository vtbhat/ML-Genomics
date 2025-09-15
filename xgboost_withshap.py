import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV

import shap

##-----Read the data
genes_df = pd.read_csv("/geneticsvariants_UC.csv")


##-----Remove genes with no variants
genes_df = genes_df.loc[:, (genes_df != 0).any(axis=0)]

##----Read in the phenotype data
phenotype_df = pd.read_csv("/IBDcovariates.covar.txt", sep='\t')

print("Finished reaidng files")

##-----Merge the two dataframes
merged_df = pd.merge(genes_df, phenotype_df, how="outer", on="IID")

##----Optional: Drop unnecessary columns
merged_df = merged_df.drop(['PID', 'FID', 'MID'], axis=1)


print(merged_df.head())
print(merged_df.shape)

##-----Declare dependent and indepdendent variables
X = merged_df.drop(['disease', 'IID'], axis=1)
y = merged_df['disease']

##-----Train the model
data_dmatrix = xgb.DMatrix(data=X,label=y)

##-----Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np

##-----Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

##---Base parameters
params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',     # fast histogram algorithm
    'eval_metric': 'auc',
    'n_estimators': 800       # let early stopping decide
}

xgb_clf = XGBClassifier(**params)

##-----Search space for hyperparameter tuning
param_dist = {
    'learning_rate': [0.05, 0.1, 0.2, 0.5],
    'max_depth': [3,4,5],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'alpha': [0.1, 0.5, 1, 5],
     'early_stopping_rounds': [10]
}

##-----Randomized Search
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=30,                # sample 30 combinations
    scoring='roc_auc',
    cv=3,                     # 3-fold CV (faster for big data)
    verbose=1,
    n_jobs=-1,
    random_state=0
)

##---Fit with early stopping
random_search.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

##-----Best model
best_xgb = random_search.best_estimator_

print("Best parameters:", random_search.best_params_)
print("Best CV score:", random_search.best_score_)

##-----Test performance
y_pred = best_xgb.predict(X_test)
print("Accuracy: %.4f" % accuracy_score(y_test, y_pred))
print("ROC AUC: %.4f" % roc_auc_score(y_test, y_pred))
print("PR AUC: %.4f" % average_precision_score(y_test, y_pred))
best_params = random_search.best_params_


##----Train the XGBoost model for SHAP
model = xgb.XGBClassifier(
   max_depth=best_params['max_depth'], 
   learning_rate=best_params['learning_rate'], 
   alpha=best_params['alpha'], 
   subsample=best_params['subsample'],  
   colsample_bytree=best_params['colsample_bytree'], 
   n_estimators=800, 
   min_child_weight=best_params['min_child_weight'], 
   objective='binary:logistic')

model.fit(X_train, y_train)

##---Create a SHAP explainer tree
explainer = shap.TreeExplainer(model)

##-----Computed SHAP values
shap_values = explainer.shap_values(X_test)

##-----Createa matplot;lib amtplotlib plot for storing the SHAP summary plot
fig, ax = plt.subplots(figsize=(10, 8))  # Optional: adjust the figure size as needed

##----Generate the SHAP summary plot (this will use the current axes `ax`)
shap.summary_plot(shap_values, X_test, show=False)  # `show=False` prevents automatic display

##----Save the plot to a file
plt.savefig('/shap_summary_plotUC.png')

##---Close the plot
plt.close(fig)
