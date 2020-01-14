#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:59:55 2020

@author: Flavia García Vázquez
"""


from TSVM import TSVM
import numpy as np

import sys
sys.path.insert(1, '../clustering')

import digit_datasets 

# UNBALANCED DATA 
#numbers_train = pd.read_csv("numbers_train.csv", sep= " ", header = None).iloc[:,:257]
#training_samples = numbers_train.sample(2000, random_state = 123)
#
#features_training_data = training_samples.drop(0, axis = 1)
#targets_training_data = training_samples.loc[:,0]
#
#targets_training_data_bin = (targets_training_data>4).astype(int)
#targets_training_data_bin[targets_training_data_bin==0] = -1


features_training_data,targets_training_data = digit_datasets.generateBalancedDataset()

targets_training_data[targets_training_data==0] = -1 #The targets have to be +1 or -1

n_subsets = 50
n_train_samples = 2000

n_samples_cluster = n_train_samples/n_subsets


error_each_fold = []


## RBF kernel
weight_rbf = 5

model = TSVM()
model.initial('rbf', gamma=weight_rbf, C_labeled=10, C_unlabeled=1, num_positives = 2)
    

## CROSS VALIDATION C
for k_fold in range(1,n_subsets+1):
    lower_limit = int((k_fold - 1) * n_samples_cluster)
    upper_limit = int(k_fold * n_samples_cluster)
    
    features_labeled_data = features_training_data[lower_limit:upper_limit]
    targets_labeled_data = targets_training_data[lower_limit:upper_limit]
        
    idx_to_delete = list(range(lower_limit,upper_limit))
    features_unlabeled_data = np.delete(features_training_data, idx_to_delete, axis=0)
    targets_unlabeled_data = np.delete(targets_training_data, idx_to_delete)
    
    model.train(features_labeled_data, targets_labeled_data, features_unlabeled_data)

    accuracy = model.score(features_unlabeled_data, targets_unlabeled_data)
    error = (1 - accuracy)*100
    
    print("---------------")
    print(error)
    
    error_each_fold.append(error)





