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
	
	training_samples = numbers_train.sample(3000,random_state=123)
	
	## CROSS VALIDATION C

	
	unlabeled_data = training_samples
	unlabeled_data=np.array(unlabeled_data)
	X=unlabeled_data[:,1:]
	Y=unlabeled_data[:,0]
	#Y=np.where(Y>=5,1,0)

	return X,Y
	## RBF kernel


def preProcessDigits(inputs,targets):
	idx_numbers=[]
	for i in range(10):
		tmp=np.where(targets==i)[0]
		#tmp=np.array(tmp)
		#print(tmp)
		idx_numbers.append(tmp)
	ratio=[40*len(x)/2000  for x in idx_numbers]
	#print(ratio)
	np.random.seed(123)
	X=np.zeros(2000)
	c=0
	for i in range(200):
		for j in range(10):
			idx=np.random.randint(0,len(idx_numbers[j]))
			X[c]=idx_numbers[j].take(idx)
			np.delete(idx_numbers[j],idx)
			c+=1
	#print(np.shape(inputs))
	#print(len(X))
	#print(X)
	#print(targets[0])
	out_target=np.zeros(2000)
	out_points=np.zeros(shape=(2000,256))
	c=0
	for i in X:
		out_target[c]=targets[int(i)]
		out_points[c]=inputs[int(i)]
		c+=1
	#targets[X] contains bla bla bla 
	return out_points,out_target


	
def generateBalancedDataset():
	x,y=generateDigitsDataset()
	x,y =preProcessDigits(x,y)
	return x,y
def generateUnbalancedDataset():
	x,y=generateDigitsDataset()
	y=np.where(y>=5,1,0)
	return x[0:2000,:],y[0:2000]



	
