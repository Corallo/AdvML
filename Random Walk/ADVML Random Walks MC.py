#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Digits
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
digits = datasets.load_digits()
news = datasets.fetch_20newsgroups
news_train = news(subset='train', categories= ['comp.os.ms-windows.misc', 'comp.sys.mac.hardware'])
data = digits.data
target_train = digits.target


# In[265]:


# #titanic
# from sklearn import datasets
# from sklearn import metrics
# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# import random
# import pandas as pd 
# data_train = pd.read_csv('titanic_data_train.csv', sep=',')
# target_train = pd.read_csv('titanic_labels_train.csv', sep=',')
# iris = datasets.load_iris()
# digits = datasets.load_digits()
# news = datasets.fetch_20newsgroups
# news_train = news(subset='train', categories= ['comp.os.ms-windows.misc', 'comp.sys.mac.hardware'])
# data = np.array(data_train)
# target_train = np.array(target_train)


# In[266]:


# #heart disease
# from sklearn import datasets
# from sklearn import metrics
# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# import random
# import pandas as pd 
# data_train = pd.read_csv('heart_data.csv', sep=',')
# target_train = pd.read_csv('heart_target.csv', sep=',')
# iris = datasets.load_iris()
# digits = datasets.load_digits()
# news = datasets.fetch_20newsgroups
# news_train = news(subset='train', categories= ['comp.os.ms-windows.misc', 'comp.sys.mac.hardware'])
# data = np.array(data_train)
# target_train = np.array(target_train)


# In[284]:


# #wine
# from sklearn import datasets
# from sklearn import metrics
# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# import random
# import pandas as pd 
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# import matplotlib.pyplot as plt
# import progressbar
# from sklearn.cluster import KMeans

# data_train = pd.read_csv('wine_data.csv', sep=',')
# target_train = pd.read_csv('wine_target.csv', sep=',')
# iris = datasets.load_iris()
# digits = datasets.load_digits()
# news = datasets.fetch_20newsgroups
# news_train = news(subset='train', categories= ['comp.os.ms-windows.misc', 'comp.sys.mac.hardware'])
# data = np.array(data_train)
# target_train = np.array(target_train)


# In[268]:


# #iris
# from sklearn import datasets
# from sklearn import metrics
# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# import random
# import pandas as pd 
# data_train = pd.read_csv('wine_data.csv', sep=',')
# target_train = pd.read_csv('wine_target.csv', sep=',')
# iris = datasets.load_iris()
# digits = datasets.load_digits()
# news = datasets.fetch_20newsgroups
# news_train = news(subset='train', categories= ['comp.os.ms-windows.misc', 'comp.sys.mac.hardware'])
# data = np.array(iris.data)
# target_train = np.array(iris.target)


# In[4]:


#calculate pairwise distance #https://papers.nips.cc/paper/1896-data-clustering-by-markovian-relaxation-and-the-information-bottleneck-method.pdf
def distance(data):
    distance_matrix = metrics.pairwise_distances(data)
    return distance_matrix


# In[5]:


#find indices of nearest neighbors
def knearest(data,k):
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(data)
    distances, indices = neighbors.kneighbors(data)
    return neighbors.kneighbors(data)


# In[6]:


#Assign weights to nearest neighbors
def weights(data,d,indices,sigma):
    weights = np.zeros((data.shape[0],data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(indices.shape[1]):
            weights[indices[i][j],i]=np.exp(-d[indices[i][j],i]/sigma**2)
            weights[i,i]=1
    return weights


# In[7]:


#one step transition matrix
def p_one(weights):
    num = np.zeros((weights.shape))
    denom = np.zeros(weights.shape[0])
    p = np.zeros((weights.shape))
    for i in range(p.shape[0]):
        denom[i] = np.sum(weights[i,:])
        for j in range(p.shape[0]):
            num[i,j] = weights[i,j]
            if denom[i] != 0:
                p[i,j] = num[i,j]/denom[i]
    return p


# In[8]:


#t-step transition matrix
def p_t(p_one_m,t):
    return np.linalg.matrix_power(p_one_m,10)


# In[9]:


#E-step
def E_step(p_t_matrix,P_y):
    return p_t_matrix*P_y


# In[55]:


#M-step
def M_step(e_step,p_t_matrix):
    denom = np.zeros(p_t_matrix.shape[1])
    num = np.zeros(e_step.shape[1])
    for i in range(p_t_matrix.shape[1]):
        denom[i] = np.sum(p_t_matrix[:,i])
        num[i] = np.sum(e_step[:,i])
    P_y=np.zeros((e_step.shape[1],1))   
    P_y[:,0]=num/denom
    return P_y


# In[11]:


#Loglikelihood
def likelihood(p_t_matrix,P_y):
    N_sum = np.zeros(p_t_matrix.shape[1])
    #print(p_t_matrix.shape[1])
    for i in range(p_t_matrix.shape[1]):
        N_sum[i] = np.sum(p_t_matrix[:,i]*P_y[:])
    log_N_sum = np.sum(np.log(N_sum))
    return log_N_sum  


# In[12]:


def random_walk(data,target,kneighbors=5,sigma=1, t=3, max_iter=50):#pairwise distance
    d = distance(data)
    #indices from k-nearest neighbor
    distances, indices = knearest(data,kneighbors)
    #find weights
    Weights = weights(data,d,indices,sigma)
    #one step transition probability matrix
    p_one_matrix = p_one(Weights)
    #t step transition probability matrix
    p_t_matrix = p_t(p_one_matrix,t)

    P_y=np.zeros((data.shape[0],1))
    for i in range(len(target)):
        P_y[i,0] = target[i]/1

    print("P_y =" ,P_y)
    likelihood_new = 1
    likelihood_old = 0

    
    #iterate until loglikelihood no longer improves
    iteration = 0
    #while (abs(likelihood_new-likelihood_old) > .0003):
    for i in range(max_iter):
        iteration += 1
        print("iteration = ",iteration)
        #likelihood_old = likelihood_new
        e_step=E_step(p_t_matrix,P_y)
        print("finish e")
        P_y = M_step(e_step,p_t_matrix)
        print("P_y sum = ",np.sum(P_y))
        print("P_y = ",P_y)
        print("finish m")
        #likelihood_new = likelihood(p_t_matrix,P_y)
        print("finish likelihood")
        #print("likelihood = ",likelihood_new)

    predictions = np.zeros(data.shape[0])
    #make classification predictions
    for i in range(data.shape[0]):
        if P_y[i] >= 0.5:
            predictions[i] = 1
        if P_y[i] < 0.5:
            predictions[i] = 0
    print(np.sum(predictions))
    print(predictions)
    return predictions


# In[90]:


def generateSubset(inputs, targets, x):
    train_idx = range(40 * (x - 1), 40 * x)
    target = np.full((len(targets)),0.5)
    target[train_idx] = targets[train_idx]

    return inputs, target, train_idx


# In[ ]:


def generateMulticlassSubset(inputs, targets, x):
    train_idx = range(40 * (x - 1), 40 * x)
    target = np.full((len(targets)),0.5)
    target[train_idx] = targets[train_idx]

    return inputs, target, train_idx


# In[ ]:


def DigitTestBinary(inputs, targets):
    print("Testing...")
    original_score=np.zeros(50)
    kernel_score=np.zeros(50)
    for x in progressbar.progressbar(range(50)):
        #print(x)
        actual_targets = targets.copy()
        print(np.sum(actual_targets))
        inputs, target_sample, label_idx = generateSubset(inputs,targets,x+1)
        #original = svm.SVC(kernel='linear').fit(train_points, train_targets)
        #original_score[x]=original.score(test_points, test_targets)
        new = random_walk(inputs,target_sample,kneighbors=5,sigma=1,t=3,max_iter=3)
        print(np.sum(new))
        kernel_score[x] = metrics.mean_squared_error(np.delete(new.round(),label_idx,0), np.delete(actual_targets.round(),label_idx,0))
        print(kernel_score[x])
    print("\n")

    #print("Average score of normal SVM:")
    #print(np.average(original_score))
    print("Average score of cluster kernel:")
    print(np.average(kernel_score))
    return kernel_score


# In[156]:


def DigitTestMulticlass(inputs, targets):
    print("Testing...")
    original_score=np.zeros(50)
    kernel_score=np.zeros(50)
    actual_targets = targets.copy()
    classes = len(np.unique(targets))
    for x in progressbar.progressbar(range(50)):
        prediction_matrix = np.zeros((classes,len(targets)))
        inputs, target_sample, label_idx = generateSubset(inputs,targets,x+1)
        for i in range(10):
            print(i," vs all")
            target = np.full((len(target_sample)),0.5)
            for j in label_idx:
                if actual_targets[j] == i:
                    target[j] = 1.
                else:
                    target[j] = 0.
            prediction_matrix[i,:] = random_walk(inputs,target,kneighbors=30,sigma=1,t=3,max_iter=3)
        results = np.zeros(target.shape)
        for i in range(target.shape[0]):
            if (np.sum(prediction_matrix[:,i]) == 1):
                results[i] = np.where(prediction_matrix[:,i]==1)[0] + int(np.min(np.unique(actual_target))) 
        kernel_score[x] = metrics.accuracy_score(np.delete(results.round(),label_idx,0), np.delete(actual_targets.round(),label_idx,0))
        print("accuracy = ",kernel_score[x])
    print("\n")

    #print("Average score of normal SVM:")
    #print(np.average(original_score))
    print("Average score of cluster kernel:")
    print(np.average(kernel_score))
    return kernel_score


# In[157]:


inputs,targets=generateBalancedDataset()
DigitTestMulticlass(inputs,targets)


# In[158]:


# classes = len(np.unique(target_train))
# actual_target = np.array(target_train)
# prediction_matrix = np.zeros((classes,len(targets)))

# for i in range(int(np.min(target_train)),int(np.max(target_train))+1):
#     target0 = np.zeros((target_train.shape))
#     print(i," vs all")
#     for j in range(len(actual_target)):
#         if actual_target[j] == i:
#             target0[j] = 1.
#         else:
#             target0[j] = 0.
#     target = np.array((target0))
#     for t in range(len(target0)):
#         if (t%2) == 0:
#             target[t] = .5
            
#     prediction_matrix[i-int(np.min(inputs)),:] = random_walk(inputs,targets,kneighbors=5,sigma=1,t=3,max_iter=1)
    


# In[159]:


# for i in range(len(target)):
#     print(np.sum(prediction_matrix[:,i]))
# results = np.zeros(target.shape)
# for i in range(target.shape[0]):
#     if (np.sum(prediction_matrix[:,i]) == 1):
#         results[i] = np.where(prediction_matrix[:,i]==1)[0] + int(np.min(np.unique(actual_target)))
# print(np.where(prediction_matrix[:,i]==1)[0])
        
# print(results)
# print(actual_target)
# print(metrics.accuracy_score(actual_target, results))

