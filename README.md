## Analysis of Genomic Data using Machine Learning: Looking at Genomic Data through a New Lens

This repository contains scripts to apply machine learning models to analyze genomic and transcriptomic data and extract insights useful for biomarker discovery

Project 1: Identification of Genes Contributing to Disease Diagnosis
Here, we provide a SamplexGene matrix as input (with each cell being the number of SNPs for that gene in a particular sample), and implement an XGBoost model to examine the accuracy of the model in classifying the samples as disease/healthy. Then, we apply SHAP to narrow down the genes that contribute the most to class prediction. You can also combine this with principal compoenents and other covariates (age, sex,e tc).
Sample input:
\begin{table}[]
\begin{tabular}{llllllll}
IID      & PTPN22 & FCGR2A & BTNL2 & C1orf141 & GPR35 & TYK2 & MST1 \\
ID\_1023 & 2      & 3      & 12    & 2        & 2     & 2    & 4    \\
ID\_1456 & 1      & 1      & 0     & 5        & 5     & 0    & 3    \\
ID\_1915 & 2      & 1      & 11    & 2        & 2     & 1    & 1    \\
ID\_2134 & 1      & 4      & 1     & 2        & 3     & 1    & 3   
\end{tabular}
\end{table}
