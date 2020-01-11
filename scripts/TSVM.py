#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:56:35 2020

@author: flaviagv
"""

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np


NB_SAMPLES = 500
NB_UNLABELED = 400
NB_LABELED = NB_SAMPLES - NB_UNLABELED

C_LABELED = 1.0
C_UNLABELED = 10.0  # We want to penalize more the missclassification of unlabeled samples 
    



def svm_target(theta, X, Y):
    """
    Objective function to optimize (OP3)
    """
    input_dimensions = X.shape[1]
    wt = theta[0:input_dimensions].reshape((input_dimensions, 1))
    
    s_eta_labeled = np.sum(theta[input_dimensions:input_dimensions + NB_LABELED])
    s_eta_unlabeled = np.sum(theta[input_dimensions + NB_LABELED:input_dimensions + NB_SAMPLES])
    
    return (C_LABELED * s_eta_labeled) + (C_UNLABELED * s_eta_unlabeled) + (0.5 * np.dot(wt.T, wt))



def labeled_constraint(theta, X, Y, idx):
    """
    First constraint OP3
    """
    input_dimensions = X.shape[1]
    wt = theta[0:input_dimensions].reshape((input_dimensions, 1))
    b = theta[-1]
    eta_labeled = theta[input_dimensions:input_dimensions + NB_LABELED]
    c = Y[idx] * (np.dot(X[idx], wt) + b) + eta_labeled[idx] - 1.0
    
    return (c >= 0)[0]


def unlabeled_constraint(theta, X, idx):
    """
    Second constraint OP3
    """
    input_dimensions = X.shape[1]
    wt = theta[0:input_dimensions].reshape((input_dimensions, 1))
    b = theta[-1]
    eta_unlabeled = theta[input_dimensions + NB_LABELED:input_dimensions + NB_SAMPLES]
    y_unlabeled = theta[input_dimensions + NB_SAMPLES:input_dimensions + NB_SAMPLES + NB_UNLABELED]
    
    c = y_unlabeled[idx - NB_LABELED] * \
        (np.dot(X[idx], wt) + b) + \
        eta_unlabeled[idx - NB_LABELED] - 1.0
    
    return (c >= 0)[0]

def eta_labeled_constraint(theta, X, idx):
    """
    eta has to be greater or equal to 0 
    """
    input_dimensions = X.shape[1]
    eta_labeled = theta[input_dimensions:input_dimensions + NB_LABELED]
    return eta_labeled[idx] >= 0

def eta_unlabeled_constraint(theta, X, idx):
    """
    eta has to be greater or equal to 0 
    """
    input_dimensions = X.shape[1]
    eta_unlabeled = theta[input_dimensions + NB_LABELED:input_dimensions + NB_SAMPLES]
    return eta_unlabeled[idx - NB_LABELED] >= 0

def y_constraint(theta, X, idx):
    """
    y has to be +1 or -1
    """
    input_dimensions = X.shape[1]
    y_unlabeled = theta[input_dimensions + NB_SAMPLES:input_dimensions + NB_SAMPLES + NB_UNLABELED]
    return np.power(y_unlabeled[idx], 2) == 1.0



def get_svm_constraints(X, Y):
    svm_constraints = []
    
    for i in range(NB_LABELED):
        svm_constraints.append({
                'type': 'ineq',
                'fun': labeled_constraint,
                'args': (X, Y, i)
            })
        svm_constraints.append({
                'type': 'ineq',
                'fun': eta_labeled_constraint,
                'args': (X, i)
            })
        
    for i in range(NB_LABELED, NB_SAMPLES):
        svm_constraints.append({
                'type': 'ineq',
                'fun': unlabeled_constraint,
                'args': (X, i)
            })
        svm_constraints.append({
                'type': 'ineq',
                'fun': eta_unlabeled_constraint,
                'args': (X, i)
            })
    
    for i in range(NB_UNLABELED):
        svm_constraints.append({
                'type': 'eq',
                'fun': y_constraint,
                'args': (X, i)
            })
        
    return svm_constraints
    

def initialize_parameters(X):
    w = np.random.uniform(-0.1, 0.1, size=X.shape[1])
    eta_labeled = np.random.uniform(0.0, 0.1, size=NB_LABELED)
    eta_unlabeled = np.random.uniform(0.0, 0.1, size=NB_UNLABELED)
    y_unlabeled = np.random.uniform(-1.0, 1.0, size=NB_UNLABELED)
    b = np.random.uniform(-0.1, 0.1, size=1)
       
    theta0 = np.hstack((w, eta_labeled, eta_unlabeled, y_unlabeled, b))
    
    return theta0

    
def load_data(): 
    X, Y = make_classification(n_samples=NB_SAMPLES, n_features=2, n_redundant=0, random_state=1000)
    Y[Y==0] = -1
    Y[NB_SAMPLES - NB_UNLABELED:NB_SAMPLES] = 0
    return X,Y




if __name__ == "__main__":
    X,Y = load_data()
    plt.scatter(X[:,0],X[:,1], c= Y, alpha = 0.2)

    
    theta0 = initialize_parameters(X)
    
    svm_constraints = get_svm_constraints(X, Y)
    
    result = minimize(fun=svm_target, 
                      x0=theta0, 
                      constraints=svm_constraints, 
                      args=(X, Y), 
                      method='SLSQP', 
                      tol=0.0001, 
                      options={'maxiter': 1000})


    # Compute the labels of the unlabeled data 
    theta_end = result['x']
    w = theta_end[0:2]
    b = theta_end[-1]
    
    Xu= X[NB_SAMPLES -NB_UNLABELED:NB_SAMPLES]
    yu = -np.sign(np.dot(Xu, w) + b)
    print("Hello")
    
    
    