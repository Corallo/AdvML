#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:59:55 2020

@author: flaviagv
"""


import pandas as pd

numbers_train = pd.read_csv("numbers_train.csv", sep= " ", header = None).iloc[:,:256]

n_train_samples = 2000
n_subsets = 50
n_samples_cluster = n_train_samples/n_subsets

training_samples = numbers_train.sample(2000)

## CROSS VALIDATION C
k_fold = 2

lower_limit = int((k_fold - 1) * n_samples_cluster)
upper_limit = int(k_fold * n_samples_cluster)

labeled_data = training_samples[lower_limit:upper_limit]

indexes_to_delete = training_samples.index[lower_limit:upper_limit]


unlabeled_data = training_samples.drop(indexes_to_delete)

## RBF kernel
weight_rbf = 5