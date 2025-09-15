## Analysis of Genomic Data using Machine Learning: Looking at Genomic Data through a New Lens

This repository contains scripts to apply machine learning models to analyze genomic and transcriptomic data and extract insights useful for biomarker discovery

###### Project 1: Identification of Genes Contributing to Disease Diagnosis
Here, we provide a SamplexGene matrix as input (with each cell being the number of SNPs for that gene in a particular sample), and implement an XGBoost model to examine the accuracy of the model in classifying the samples as disease/healthy. Then, we apply SHAP to narrow down the genes that contribute the most to class prediction. You can also combine this with principal components and other covariates (age, sex, etc.).

###### Sample input:


| IID           | PTPN22        | FCGR2A | BTNl2 | GPR35 |
| ------------- |:-------------:| ------:|------:|------:|
| ID_1023       | 2      | 3    | 12 | 2 |
| ID_1856       | 1      |   1  | 0  | 5 |
| id_3457       | 2      |    1  | 11 | 2 |

###### Sample output:
The prediction task here is classifying whether a sample is healthy or has ulcerative colitis (UC), a type of inflammatory bowel disease. This disease has low heritability (between 15 to 20%). Hence, the performance of the model is quite modest:
Best CV score: 0.6658399496778685
Accuracy: 0.6371 
ROC AUC: 0.6145
PR AUC: 0.6272

###### SHAP summary plot:
The SHAP summary plot explains the contributions of the features to the prediction. Here, the genes BTNL2, ATXN2, and the first five principal components appear to be the most informative for prediction.
<img width="600" height="750" alt="shap_summary_plotIBD_PCs1to5" src="https://github.com/user-attachments/assets/dbd8b250-1f63-48fa-8d58-67fd0e0ec74c" />

