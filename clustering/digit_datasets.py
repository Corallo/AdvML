#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:59:55 2020

@author: flaviagv
"""


import pandas as pd
import numpy as np
def generateDigitsDataset():
	numbers_train = pd.read_csv("numbers_train.csv", sep= " ", header = None).iloc[:,:257]
	
	training_samples = numbers_train.sample(2000)
	
	## CROSS VALIDATION C

	
	unlabeled_data = training_samples
	unlabeled_data=np.array(unlabeled_data)
	X=unlabeled_data[:,1:]
	Y=unlabeled_data[:,0]
	Y=np.where(Y>=5,1,0)

	return X,Y
	## RBF kernel


	
