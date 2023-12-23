## COMP 8740 Final Project: Survival prediction based on Genetic expression

Dataset for this study can be found here from cBioPortal:

Requisites:
- Download the dataset from: 
- Extract to a folder called data

data_collection.ipynb:
- Collect the data from txt files in side /data/brca_metabric
- Create merged_mrna_<>.csv inside ./data

data_preprocessing.ipynb:
- Read merged_mrna_<>.csv
- Perform: cleaning (drop nan values), scaling, and divide OS_MONTHS into sub-intervals.

filtering_feature.ipynb:
- Filtering top 500, 1000 genes based on MRMR or Info gain. This will reduce the search space for the following steps.

Wrapper methods: forward_elemination.ipynb - Currently 15 genes can achieve 85% accuracy.
- From the top genes selected, perform elemination with the aim to achieve smaller set of features and not lowering the accuracy.

Genetic algorithm: Select feature from top 500 by RandomForest with 5 classifier for evaluation.


k_folds.py:
- Cross validate
- Apply SMOTE oversampling on the training test of each fold.
- Output the accuracy for each fold.

Other <method>_feature_selection.ipynb: 
- Perform feature selection, store the result in txt files (if applicable). 
- Train a classifier on the selected features.
