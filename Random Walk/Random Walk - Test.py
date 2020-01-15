#!/usr/bin/env python
# coding: utf-8

# In[32]:


from new_dataset import get_20newsgroup_tf_idf
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import progressbar
from sklearn.cluster import KMeans
from digit_datasets import generateDigitsDataset, generateBalancedDataset, generateUnbalancedDataset
from ADVML_RW_MC import random_walk


# In[72]:


def test_different_trials(inputs, targets, n_classes=2, n_trials=20, n_labeled_points=2):
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
        
        target = targets.copy()
        target = np.full((len(targets)),0.5)
        target[idx_labeled_points] = targets[idx_labeled_points][0]

        # TODO: change here the function that calls your algorithm 
        classifications = random_walk(inputs,target,kneighbors=10,sigma=.6,t=8,max_iter=15)
        accuracy = metrics.accuracy_score(targets,classifications)
        
        print("Trial " + str(n_trial) + ": accuracy " + str(accuracy) + " %")
        error_trials.append(accuracy)
        
    return error_trials


# In[73]:


inputs, targets = get_20newsgroup_tf_idf("all", ["comp.windows.x", "comp.sys.mac.hardware"], 7511)
#inputs = pd.read_csv('heart_data.csv', sep=',')
#targets = pd.read_csv('heart_target.csv', sep=',')
inputs = np.array(inputs)
targets = np.array(targets)

accuracy_trials = test_different_trials(inputs, targets)
print(accuracy_trials)
accuracy_trials = np.array(accuracy_trials)

print("Average: " + str(np.average(accuracy_trials)))
print("Standard deviation: " + str(np.std(accuracy_trials*100)))


# In[66]:


# x = [54.2,55.4,58.9,65.4,72.9,80.6,87.14]
# for i in range(len(x)):
#     x[i] = 100-x[i]
# y = ["2 pts", "4 pts", "8 pts", "16 pts", "32 pts", "64 pts", "128 pts"]
# plt.title("RVM Error Rate")
# plt.plot(y,x,linestyle='--', marker='o', color='b', label = "Average Std = .610")
# plt.xlabel('Number of Labeled Points')
# plt.ylabel('Error Rate (%)')
# plt.legend()
# plt.show()


# In[71]:


# 49.12 --> 4 labeled points
# 49.12--> 6 labeled points
# 49.22 —> 8 labeled points
# 49.31 —> 16 labeled points
# 50.76 —> 32 labeled points
# 50.80 —> 64 labeled points



# x = [49.12,49.12,49.22,49.31,50.76,50.80]
# for i in range(len(x)):
#     x[i] = 100-x[i]
# y = ["2 pts", "4 pts", "8 pts", "16 pts", "32 pts", "64 pts"]
# plt.title("TSVM Error Rate")
# plt.plot(y,x,linestyle='--', marker='o', color='b', label = "Average Std = .610")
# plt.xlabel('Number of Labeled Points')
# plt.ylabel('Error Rate (%)')
# plt.ylim((10,55))
# #plt.legend()
# plt.show()


# In[ ]:




