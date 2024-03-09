# DMIE (discriminative mutual information estimation)

This is a new modeling approach for dynamic feature selection by estimating the conditional mutual information in a discriminative fashion. The implementation was done using [PyTorch Lightning](https://www.pytorchlightning.ai/index.html).

## cmi_estimator.py
For estimating Conditional Mutual Information (CMI) and performing feature selection

## data_utils.py
For loading our HEV dataset and other related necessary functions included

## greedy_ptl.py
For training and evaluating the Greedy Dynamic Selection model

## masking_pretrainer.py
For pretraining a model with missing features
