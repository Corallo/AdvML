import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigs
from sklearn.datasets import make_circles
from sklearn import svm
#from digit_datasets import *
from new_dataset import get_20newsgroup_tf_idf
import progressbar
import time
from sklearn.cluster import KMeans


def radialBasisFunctionKernel(x, y, sigma=0.55):
    return np.exp(-np.linalg.norm(x - y) / (2 * sigma**2))


def loadDataset(fileX, fileY):
    x = np.genfromtxt(fileX, delimiter=',')
    y = np.genfromtxt(fileY, delimiter=',', dtype=np.int) - 1
#	y=np.array([k if k == 1 else -1 for k in y])
    return x, y


def randomize(inputs, targets):
    N = inputs.shape[0]  # Number  of  rows  ( s a m p l e s )
    permute = list(range(N))
    random.shuffle(permute)
    inputs_out = inputs[permute, :]
    targets_out = targets[permute]
    return inputs_out, targets_out


def generateDataset():
    np.random.seed(100)
    classA = np. concatenate(
        (	np.random.randn(3, 2) * 0.2 + [-1.5, 0.5],
          np.random.randn(3, 2) * 0.2 + [.75, -1.5])
    )

    classB = np. concatenate(
        (
            np.random.randn(3, 2) * 0.2 + [2.5, 0.5],
        )
    )

    inputs = np.concatenate(
        (classA, classB)
    )
    targets = np.concatenate(
        (np.ones(classA.shape[0]),
         -np.ones(classB.shape[0])))
    return inputs, targets


def TestResults(inputs, targets, results, iterations):
    train_points, test_points = partition(inputs, 0.7)
    train_targets, test_targets = partition(targets, 0.7)
    kernel_train, kernel_test = partition(results, 0.7)

    print("Testing...")

    print("Learning model without kernel")
    original = LogisticRegression(max_iter=iterations).fit(
        train_points, train_targets)
    print("Score:")
    print(original.score(test_points, test_targets))

    print("Learning model with kernel")
    new = LogisticRegression(max_iter=iterations).fit(
        kernel_train, train_targets)
    print("Score:")
    print(new.score(kernel_test, test_targets))

   # print("Predicion output with post-kernel features:")
    #prediction = new.predict(kernel_test)
    #print(prediction)
    #print("Correct output:")
    #print(test_targets)

    return (original.predict(inputs), new.predict(results))


def generateAffinityMatrix(inputs):
    N = inputs.shape[0]
    affinity_matrix = np.ones((N, N))

    for idx_row in range(N):
        for idx_col in range(N):
            if idx_row != idx_col:
                affinity_matrix[idx_row, idx_col] = radialBasisFunctionKernel(
                    inputs[idx_row], inputs[idx_col])

    return affinity_matrix


def generateDMatrix(K):
    N = len(K)
    D = np.zeros((N, N))
    np.fill_diagonal(D, np.sum(K, axis=0))
    return D


def generateLMatrix(K, D):
    D_ = fractional_matrix_power(D, -0.5)
    L = D_ @ K @ D_  # Instead of multiplication
    return L


def getEigenvectorMatrix(L, k):
    w, V = eigs(L)
    V_filtered = V[:, :k]
    return V_filtered.real


def normalizeRow(eigen_vectors):
    normalizer = np.linalg.norm(eigen_vectors, axis=1)  # one value per row
    eigen_vectors_norm = eigen_vectors / normalizer[:, None]
    return eigen_vectors_norm


def partition(X, fraction):
    breakPoint = int(len(X) * fraction)
    return X[:breakPoint], X[breakPoint:]


def plotOutput(inputs, prediction):
    red = []
    blue = []

    for c in prediction:
        idx_red = np.where(prediction == 0)[0]
        idx_blue = np.where(prediction == 1)[0]

    red = inputs[idx_red, :]
    blue = inputs[idx_blue, :]

    plt.figure()
    plt.plot(red[:, 0], red[:, 1], 'rs', blue[:, 0], blue[:, 1], 'bs')


def transfer_function(L, k, function, param = []):
    t = 2
    w, V = np.linalg.eig(L)
    w = np.diag(w)
    if function == "linear":
        return w
    if function == "step":
        r = param[0]
        w_diagonal = np.diagonal(w)
        idx_greatest_w = np.argsort(w_diagonal)[-r:]
        for i in range(w.shape[0]):
                if i in idx_greatest_w.tolist():
                    w[i, i] = 1
                else:
                    w[i, i] = 0

    if function == "linear step":
        r = param[0]
        w_diagonal = np.diagonal(w)
        idx_greatest_w = np.argsort(w_diagonal)[-r:]
        for i in range(w.shape[0]):
                if i in idx_greatest_w.tolist():
                    w[i, i] = w[i, i]
                else:
                    w[i, i] = 0

    if function == "polynomial":
        w, V = np.linalg.eig(L)
        w = np.diag(w)
        w = w**t
    if function == "polystep":
        p = 1/2
        q = 2
        r = param[0]
        w_diagonal = np.diagonal(w)

        idx_greatest_w = np.argsort(w_diagonal)[-r:]
        for i in range(w.shape[0]):
            if i in idx_greatest_w.tolist():
                w[i, i] = w[i, i]**p
            else:
                w[i, i] = w[i, i]**q

    return w


def transformL(L, transfer_function_name, param=[]):
    w, U = np.linalg.eig(L)
    L_new =  U.dot(transfer_function(L,k,transfer_function_name, param = param)).dot(np.linalg.inv(U))
    return L_new.real

def transformDMatrix(L, L_new):
	D_new = np.zeros((L.shape))
	for i in range(D_new.shape[0]):
		D_new[i,i] = 1/L_new[i,i]
	return D_new

def transformKMatrix(D_new,L_new):
	K_new = fractional_matrix_power(D_new,0.5).dot(L_new).dot(fractional_matrix_power(D_new,0.5))
	return K_new

def generateSubset(inputs, targets, kernel, x):
    test_idx = range(40 * (x - 1), 40 * x)

    train_input = inputs[test_idx, :]
    test_input  = np.delete(inputs,test_idx,0)
    
    train_targets = targets[test_idx]
    test_targets= np.delete(targets,test_idx,0)
    

    train_kernel = np.take(kernel,test_idx,axis=0)
    train_kernel = np.take(train_kernel,test_idx,axis=1)

    test_kernel = np.delete(kernel,test_idx,0)
    test_kernel = test_kernel[:,test_idx]

    return train_input, train_targets, train_kernel, test_input,test_targets, test_kernel

def TestWithSVM(inputs,targets,results):
    train_points, test_points = partition(inputs, 0.2)
    train_targets, test_targets = partition(targets, 0.2)
    kernel_train, kernel_test = partition(results, 0.2)

    print("Testing...")

    print("Learning model without kernel")
    original = svm.SVC(kernel='linear').fit(train_points, train_targets)
    print("Score:")
    print(original.score(test_points, test_targets))

    print("Learning model with kernel")
    new = svm.SVC(kernel='linear').fit(kernel_train, train_targets)
    print("Score:")
    print(new.score(kernel_test, test_targets))

    #print("Predicion output with post-kernel features:")
    #prediction = new.predict(kernel_test)
    #print(prediction)
    #print("Correct output:")
    #print(test_targets)

    return (original.predict(inputs), new.predict(results))

def computeScore(out,targets):
    return np.sum(np.where(out==targets,1,0))/len(targets)

def DigitTest(inputs, targets, results):
    print("Testing...")
    original_score=np.zeros(50)
    kernel_score=np.zeros(50)
    for x in progressbar.progressbar(range(50)):
        #print(x)   
        train_points, train_targets, kernel_train, test_points, test_targets, kernel_test = generateSubset(inputs,targets,results,x+1)
        original = svm.SVC(gamma=0.02).fit(train_points, train_targets)
        original_score[x]=original.score(test_points, test_targets)


        new = svm.SVC(kernel='precomputed').fit(kernel_train, train_targets)
        kernel_score[x]=new.score(kernel_test, test_targets)

    print("\n")

    print("Average score of normal SVM:")
    print(np.average(original_score))
    print("Average score of cluster kernel:")
    print(np.average(kernel_score))
    return original_score, kernel_score

def testNews(inputs, targets, kernel):
    random.seed(123)
    print("Testing..")
    print(targets)
    original_score = np.zeros(100)
    kernel_score = np.zeros(100)
    myrange=range(100,900,100) # test with different number of test_points (2,4,8,16 ..)
    output_mean=[0]*len(myrange)
    output_std=[0]*len(myrange)
    i=0
    for n in myrange:
        print("\n\nUsing "+str(n)+" train points")
        for x in range(100): #run 100 times
            target_sum=0
            while(target_sum!=n/2): #be sure that you have selected at least 1 point for each cluster
                train_idx=random.sample(range(0, int(len(targets)/2)), n)
                train_targets = np.take(targets,train_idx)
                target_sum=sum(train_targets)

            test_targets = np.delete(targets,train_idx)
            
            train_points = np.take(inputs,train_idx,axis=0)
            test_points = np.delete(inputs,train_idx,axis=0)
            
            train_kernel = np.take(kernel, train_idx,axis=0)
            train_kernel = np.take(train_kernel, train_idx, axis=1)

            test_kernel= np.delete(kernel, train_idx, axis=0)
            test_kernel= np.take(test_kernel,train_idx,axis=1)

            #original = svm.SVC(kernel='rbf',gamma=1.65).fit(train_points, train_targets)
            #original_score[x]=original.score(test_points, test_targets)


            new = svm.SVC(kernel='precomputed').fit(train_kernel, train_targets)
            kernel_score[x]=new.score(test_kernel, test_targets)


        output_mean[i]=np.average([1-x for x in kernel_score])
        output_std[i]=np.std([1-x for x in kernel_score])
        i+=1
        print("Average score of cluster kernel:")
        print(np.average(kernel_score))
    return output_mean, output_std

def psi_function(x):
    if x > 0:
        return 1
    else:
        return 0
def automatic_selection_news(inputs, target):
    print(len(inputs))
    tfs=['linear','step','polynomial','polystep']
    N = inputs.shape[0]
    #print(N)
    k = 2  # Desired NUMBER OF CLUSTERS (small k)
    print("computing K")
    K = generateAffinityMatrix(inputs)  # (uppercase K) STEP 1
    print("computing D")
    D = generateDMatrix(K)  # STEP 2
    print("computing L")
    L = generateLMatrix(K, D)
    plt.figure()
    for tf in tfs:

        print("computing L new with "+ tf)
        L_new = transformL(L, tf, param=[10])
        print("computing D new")
        D_new = transformDMatrix(L, L_new)
        print("computing K new")
        K_new = transformKMatrix(D_new,L_new)
        avg, std= testNews(inputs,target,K_new)
        
        

        plt.errorbar(range(100,900,100), avg, yerr=std)
        plt.xticks(range(100,900,100))
        plt.xscale('log', basex=2)
        plt.xlabel("Number of train points")
        plt.ylabel("Error")
    plt.legend(tfs)
    plt.show()

    #DigitTest(inputs,target,K_new)
   #     train_kernel = np.take(K_new, train_idx,axis=0)
   #     train_kernel = np.take(train_kernel, train_idx, axis=1)
#
#   #     test_kernel= np.delete(K_new, train_idx, axis=0)
#   #     test_kernel= np.take(test_kernel,train_idx,axis=1)
#
#   #     print(train_kernel.shape)
#   #     new = svm.SVC(kernel='precomputed').fit(train_kernel, train_targets)
#   #     test_errors.append(1 - (new.score(test_kernel, test_targets)))
#   #     alphas = np.abs(new.dual_coef_[0])
#   #     T = 0.0
#   #     K = np.zeros((new.support_.shape[0], new.support_.shape[0]))
#   #     for idx, x in enumerate(new.support_):
#   #         for idy, y in enumerate(new.support_):
#   #             K[idx, idy] = np.dot(inputs[x,:], inputs[y,:])
#   #     for idx, p in enumerate(new.support_):
#   #             print(alphas[idx]*K_new[p,p])
#   #             #T = T + psi_function((alphas[idx]/np.linalg.inv(K)[idx, idx]) - 1)
#   #             T = T + psi_function((alphas[idx]*K_new[p,p]) -1)
#   #             #T = T + alphas[idx]*K_new[p,p]
#   #     T = T/16
#   #     print(T)
#   #     span_estimates.append(T)
#   # X = np.arange(4, 30, 3)
#   # plt.plot(X, span_estimates)
#   # plt.plot(X, test_errors)
   # plt.show()

    return 0

#inputs, targets = make_circles(n_samples=200, shuffle=True, noise=None, random_state=150, factor=0.4)
# inputs,targets=loadDataset('irisX_small.txt','irisY_small.txt') load data from file
# inputs,targets=generateDataset()
#inputs,targets=generateDigitsDataset()

inputs,targets=get_20newsgroup_tf_idf("all", ["comp.windows.x", "comp.sys.mac.hardware"], 7511)
#inputs,targets=generateUnbalancedDataset()
#inputs,targets=generateBalancedDataset()
#inputs,targets=load_digits(n_class=10, return_X_y=True)
inputs=np.array(inputs)
targets=np.array(targets)
#inputs, targets = randomize(inputs, targets)
'''
print("Start")

N = inputs.shape[0]
#print(N)
k = 2  # Desired NUMBER OF CLUSTERS (small k)
print("computing K")
K = generateAffinityMatrix(inputs)  # (uppercase K) STEP 1
print("computing D")
D = generateDMatrix(K)  # STEP 2
print("computing L")
L = generateLMatrix(K, D)
print("computing L new")
L_new = transformL(L, "polynomial")
print("computing D new")
D_new = transformDMatrix(L)
print("computing K new")
K_new = transformKMatrix(D_new,L_new)
print("Kernel done")
'''
k = 2
automatic_selection_news(inputs, targets)
# plt.plot(V[:,0],V[:,1],'rs')

#standard_prediction, prediction = TestResults(inputs, targets, K_new, 1000)  # Test the results
#standard_prediction, prediction = TestWithSVM(inputs,targets,K_new)
#FinalTest(inputs,targets,K_new)
#testNews(inputs,targets,K_new)
#plotOutput(inputs, standard_prediction)
#plotOutput(inputs, prediction)
#plt.show()
