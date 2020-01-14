#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:40:37 2020

@author: Flavia García Vázquez
"""



import pandas as pd
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




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 20:13:51 2020

@author: flaviagv
"""


import sys
sys.path.insert(1, '../clustering')

from new_dataset import get_20newsgroup_tf_idf

sys.path.insert(1, '../scripts')

from TSVM import TSVM

import numpy as np
import random 



random.seed(123)



def test_different_trials(features, targets, n_classes=2, n_trials=100, n_labeled_points=16):
    N = len(targets)
    error_trials = []
    for n_trial in range(n_trials):
        idx_labeled_points = random.sample(range(N), n_labeled_points)
    
        targets_labeled = targets[idx_labeled_points]
        
        # If we dont have at least one element 
        while len(np.unique(targets_labeled)) != n_classes:
            print("Looking for another sample")
            idx_labeled_points = random.sample(range(N), n_labeled_points)
            targets_labeled = targets[idx_labeled_points]
        
        inputs_labeled = features[idx_labeled_points]

        targets_unlabeled = np.delete(targets, idx_labeled_points)
        inputs_unlabeled = np.delete(features, idx_labeled_points, axis = 0)

        error = train_tsvm_algorithm_bin(inputs_labeled, targets_labeled, inputs_unlabeled, targets_unlabeled)  
        
        print("Trial " + str(n_trial) + ": error " + str(error) + " %")
        error_trials.append(error)
        
    return error_trials


def train_tsvm_algorithm_bin(inputs_labeled, targets_labeled, inputs_unlabeled, targets_unlabeled, weight_rbf = 5):
    targets_labeled[targets_labeled == 0] = -1
    targets_unlabeled[targets_unlabeled == 0] = -1
    
    model = TSVM()
    
    model.initial('rbf', gamma=weight_rbf, C_labeled=10, C_unlabeled=1, num_positives = 2)
    
    model.train(inputs_labeled, targets_labeled, inputs_unlabeled)
    
    # Y_hat = model.predict(inputs_unlabeled)
    
    accuracy = model.score(inputs_unlabeled, targets_unlabeled)
    
    error = 1 - accuracy
    
    return error * 100





def load_and_regroup_wine_dataset():
    
    features_wine = pd.read_csv("../data/wine_data.csv", sep= ",", header = None).to_numpy()		
    target_wine = pd.read_csv("../data/wine_target.csv", sep= ",", header = None).iloc[:,0].to_numpy()
    
    new_target_wine = np.zeros(len(target_wine))
    
    # Class 1 will be class 6
    
    new_target_wine[target_wine == 6] = 1 
    
    # Class 2 will be class 7,8,9
    
    idx_class_2 = np.where((target_wine== 7) | (target_wine== 8) | (target_wine== 9)) 
    new_target_wine[idx_class_2] = 2
    
    return features_wine, new_target_wine
    
    


def test_different_trials_multiclass(features, targets, n_trials=100, n_labeled_points=16):
    classes = np.unique(targets)
    n_classes = len(classes)
    N = len(targets)
    
    error_trials = []
    for n_trial in range(n_trials):
        idx_labeled_points = random.sample(range(N), n_labeled_points)
    
        targets_labeled = targets[idx_labeled_points]
        
        # If we dont have at least one element 
        while len(np.unique(targets_labeled)) != n_classes:
            print("Looking for another sample")
            idx_labeled_points = random.sample(range(N), n_labeled_points)
            targets_labeled = targets[idx_labeled_points]
        
        inputs_labeled = features[idx_labeled_points]

        targets_unlabeled = np.delete(targets, idx_labeled_points)
        inputs_unlabeled = np.delete(features, idx_labeled_points, axis = 0)    
        
        assigned_classes = oneVSall(classes, inputs_labeled, targets_labeled, inputs_unlabeled)
        
        
        error = np.sum(assigned_classes != targets_unlabeled)*100/len(targets_unlabeled)
        
        print("Trial " + str(n_trial) + ": error " + str(error) + " %")
        
        error_trials.append(error)
        
    return error_trials



def oneVSall(classes, inputs_labeled, targets_labeled, inputs_unlabeled):
    n_classes = len(classes)
    new_targets_labeled = np.zeros(len(targets_labeled))
    
    prob_unl_points_each_class = np.zeros((len(inputs_unlabeled), n_classes))
    
    for i in range(n_classes):
        classifier_one_class = classes[i]
        
        new_targets_labeled[:] = -1
        new_targets_labeled[targets_labeled==classifier_one_class] = 1
        
        probability_that_class = train_tsvm_algorithm_multiclass(inputs_labeled, new_targets_labeled, inputs_unlabeled)
        prob_unl_points_each_class[:,i] = probability_that_class
    
    assigned_classes = np.argmax(prob_unl_points_each_class, axis=1)
    return assigned_classes




def train_tsvm_algorithm_multiclass(inputs_labeled, targets_labeled, inputs_unlabeled, weight_rbf = 5):
    model = TSVM()
    
    model.initial('rbf', gamma=weight_rbf, C_labeled=10, C_unlabeled=1, num_positives = 2)
    
    model.train(inputs_labeled, targets_labeled, inputs_unlabeled)
    
    
    probability_that_class = model.predict_proba(inputs_unlabeled)[:,1]
    
    return probability_that_class





if __name__ ==  "__main__":
    
    print("Heart dataset")
    features_heart = pd.read_csv("../data/heart_data.csv", sep= ",", header = None)		
    target_heart = pd.read_csv("../data/heart_target.csv", sep= ",", header = None).iloc[:,0]
    features_heart = features_heart.to_numpy()
    target_heart = target_heart.to_numpy()
    
    error_trials = test_different_trials(features_heart, target_heart)
    error_trials = np.array(error_trials)
    print("Average: " + str(np.average(error_trials)))
    print("Standard deviation: " + str(np.std(error_trials)))
    
    
    
    print("Titanic dataset")
    features_titanic = pd.read_csv("../data/titanic_data_train.csv", sep= ",", header = None)		
    target_titanic = pd.read_csv("../data/titanic_labels_train.csv", sep= ",", header = None).iloc[:,0]
    
    features_titanic = features_titanic.to_numpy()
    target_titanic = target_titanic.to_numpy()
    
    error_trials = test_different_trials(features_titanic, target_titanic)
    error_trials = np.array(error_trials)
    print("Average: " + str(np.average(error_trials)))
    print("Standard deviation: " + str(np.std(error_trials)))
    
    
    print("Wine dataset")
    features_wine, target_wine = load_and_regroup_wine_dataset()
    
    idx = random.sample(range(len(features_wine)), 2000)
    
    target_wine = target_wine[idx]
    features_wine = features_wine[idx]
    
    error_trials = test_different_trials_multiclass(features_wine, target_wine)
    error_trials = np.array(error_trials)
    print("Average: " + str(np.average(error_trials)))
    print("Standard deviation: " + str(np.std(error_trials)))
    
    
    
    
