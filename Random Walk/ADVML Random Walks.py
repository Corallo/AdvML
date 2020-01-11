#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Digits
# from sklearn import datasets
# from sklearn import metrics
# from sklearn.neighbors import NearestNeighbors
# import numpy as np
# import random
# iris = datasets.load_iris()
# digits = datasets.load_digits()
# news = datasets.fetch_20newsgroups
# news_train = news(subset='train', categories= ['comp.os.ms-windows.misc', 'comp.sys.mac.hardware'])
# data = digits.data
# actual_target = np.zeros(len(digits.target))
# for i in range(len(digits.target)):
#     if digits.target[i] <= 4:
#         actual_target[i] = 0.
#     if digits.target[i] > 4:
#         actual_target[i] = 1.
# print(actual_target[0:30])
# target = actual_target.copy()
# for i in range(len(digits.target)):
#     #if digits.target[i] == 2:
#     #    target[i] = 1.
#     if (i%3) == 0:
#         target[i] = 0.5
# print(target[0:30])


# In[2]:


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
# target = np.array(target_train)
# print(data.shape)
# actual_target = target.copy()
# for i in range(len(target)):
#     #if digits.target[i] == 2:
#     #    target[i] = 1.
#     if (i%2) == 0:
#         target[i] = 0.5
# print(target.shape)


# In[3]:


# #titanic
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
# target = np.array(target_train)
# print(data.shape)
# actual_target = target.copy()

# for i in range(len(target)):
#     #if digits.target[i] == 2:
#     #    target[i] = 1.
#     if (i%2) == 0:
#         target[i] = 0.5
# print(target.shape)


# In[4]:


#wine
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import pandas as pd 
data_train = pd.read_csv('wine_data.csv', sep=',')
target_train = pd.read_csv('wine_target.csv', sep=',')
iris = datasets.load_iris()
digits = datasets.load_digits()
news = datasets.fetch_20newsgroups
news_train = news(subset='train', categories= ['comp.os.ms-windows.misc', 'comp.sys.mac.hardware'])
data = np.array(data_train)
target = np.array(target_train)
print(data.shape)
for i in range(len(target)):
    if target[i] <= 5:
        target[i] = 0.
    if target[i] > 5:
        target[i] = 1.
actual_target = target.copy()
for i in range(len(target)):
    #if digits.target[i] == 2:
    #    target[i] = 1.
    if (i%2) == 0:
        target[i] = 0.5
print(target.shape)


# In[5]:


#calculate pairwise distance #https://papers.nips.cc/paper/1896-data-clustering-by-markovian-relaxation-and-the-information-bottleneck-method.pdf
def distance(data):
    distance_matrix = metrics.pairwise_distances(data)
    return distance_matrix


# In[6]:


#find indices of nearest neighbors
def knearest(data,k):
    neighbors = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(data)
    distances, indices = neighbors.kneighbors(data)
    return neighbors.kneighbors(data)


# In[7]:


#Assign weights to nearest neighbors
def weights(data,d,indices,sigma):
    weights = np.zeros((data.shape[0],data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(indices.shape[1]):
            weights[indices[i][j],i]=np.exp(-d[indices[i][j],i]/sigma**2)
            weights[i,i]=1
    return weights


# In[8]:


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


# In[9]:


#t-step transition matrix
def p_t(p_one_m,t):
    return np.linalg.matrix_power(p_one_m,10)


# In[10]:


#E-step
def E_step(p_t_matrix,P_y):
    return p_t_matrix*P_y


# In[11]:


#M-step
def M_step(e_step,p_t_matrix):
    denom = np.zeros(p_t_matrix.shape[1])
    num = np.zeros(e_step.shape[1])
    for i in range(p_t_matrix.shape[1]):
        denom[i] = np.sum(p_t_matrix[:,i])
        num[i] = np.sum(e_step[:,i])
    P_y=num/denom
    return P_y


# In[12]:


#Loglikelihood
def likelihood(p_t_matrix,P_y):
    N_sum = np.zeros(p_t_matrix.shape[1])
    #print(p_t_matrix.shape[1])
    for i in range(p_t_matrix.shape[1]):
        N_sum[i] = np.sum(p_t_matrix[:,i]*P_y[:])
    log_N_sum = np.sum(np.log(N_sum))
    return log_N_sum  


# In[13]:


def random_walk(data,kneighbors=5,sigma=1, t=3, max_iter=50):#pairwise distance
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
    print(np.sum(target))
    print(target.shape)
    for i in range(len(target)):
        P_y[i] = target[i]/1

    print(P_y)
    likelihood_new = 1
    likelihood_old = 0
    #iterate until loglikelihood no longer improves
    iteration = 0
    #while (abs(likelihood_new-likelihood_old) > .0003):
    for i in range(max_iter):
        iteration += 1
        print("iteration = ",iteration)
        likelihood_old = likelihood_new
        e_step=E_step(p_t_matrix,P_y)
        print("finish e")
        P_y = M_step(e_step,p_t_matrix)
        print("P = ",P_y)
        print("finish m")
        #likelihood_new = likelihood(p_t_matrix,P)
        print("finish likelihood")

    predictions = np.zeros(data.shape[0])
    #make classification predictions
    for i in range(data.shape[0]):
        if P_y[i] >= 0.5:
            predictions[i] = 1
        if P_y[i] < 0.5:
            predictions[i] = 0
    
    return metrics.accuracy_score(actual_target, predictions)


# In[14]:


random_walk(data,actual_target,kneighbors=5,sigma=1,t=3)


# In[ ]:




