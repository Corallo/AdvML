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




def test_different_trials(inputs, targets, n_classes=2, n_trials=100, n_labeled_points=16):
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
        
        inputs_labeled = inputs[idx_labeled_points]

        targets_unlabeled = np.delete(targets, idx_labeled_points)
        inputs_unlabeled = np.delete(inputs, idx_labeled_points, axis = 0)

        # TODO: change here the function that calls your algorithm 
        error = train_tsvm_algorithm(inputs_labeled, targets_labeled, inputs_unlabeled, targets_unlabeled)  
        
        print("Trial " + str(n_trial) + ": error " + str(error) + " %")
        error_trials.append(error)
        
        return error_trials


def train_tsvm_algorithm(inputs_labeled, targets_labeled, inputs_unlabeled, targets_unlabeled, weight_rbf = 5):
    targets_labeled[targets_labeled == 0] = -1
    targets_unlabeled[targets_unlabeled == 0] = -1
    
    model = TSVM()
    
    model.initial('rbf', gamma=weight_rbf, C_labeled=10, C_unlabeled=1, num_positives = 2)
    
    model.train(inputs_labeled, targets_labeled, inputs_unlabeled)
    
    # Y_hat = model.predict(inputs_unlabeled)
    
    accuracy = model.score(inputs_unlabeled, targets_unlabeled)
    
    error = 1 - accuracy
    
    return error * 100


       
        
        
if __name__ == "__main__":
    inputs, targets = get_20newsgroup_tf_idf("all", ["comp.os.ms-windows.misc", "comp.sys.mac.hardware"], 7511)

    targets = np.array(targets)
    
    error_trials = test_different_trials(inputs, targets)
    error_trials = np.array(error_trials)
    print("Average: " + str(np.average(error_trials)))
    print("Standard deviation: " + str(np.std(error_trials)))