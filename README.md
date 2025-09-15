## Analysis of Genomic Data using Machine Learning: Looking at Genomic Data through a New Lens

This repository contains scripts to apply machine learning models to analyze genomic and transcriptomic data and extract insights useful for biomarker discovery

Project 1: Identification of Genes Contributing to Disease Diagnosis
Here, we provide a SamplexGene matrix as input (with each cell being the number of SNPs for that gene in a particular sample), and implement an XGBoost model to examine the accuracy of the model in classifying the samples as disease/healthy. Then, we apply SHAP to narrow down the genes that contribute the most to class prediction. You can also combine this with principal compoenents and other covariates (age, sex,e tc).
Sample input:


| IID           | PTPN22        | FCGR2A | BTNl2 | GPR35 |
| ------------- |:-------------:| ------:|------:|------:|
| ID_1023       | 2      | 3    | 12 | 2 |
| ID_1856       | 1      |   1  | 0  | 5 |
| id_3457       | 2      |    1  | 11 | 2 |
