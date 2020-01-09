import random , math
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power

def radialBasisFunctionKernel(x,y):
	sigma=4
	X=np.array(x)
	Y=np.array(y)
	return np.exp(- np.linalg.norm(X-Y)/(2*sigma**2))

def loadDataset(fileX, fileY):
	x = np.genfromtxt(fileX, delimiter=',')
	y = np.genfromtxt(fileY, delimiter=',',dtype=np.int)-1
	#y=np.array([k if k == 1 else -1 for k in y])
	return x,y


def randomize(inputs, targets):
	N = inputs.shape[0] # Number  of  rows  ( s a m p l e s )
	permute=list(range(N))
	random.shuffle(permute)
	inputs_out = inputs [permute, : ]
	targets_out = targets [ permute ]
	return inputs_out, targets_out

def generateDataset():
	np.random.seed(100)
	classA = np. concatenate (
		(	np.random.randn(3, 2) * 0.2 + [-1.5, 0.5],
			np.random.randn(3, 2) * 0.2 + [.75, -1.5])
	)

	classB = np. concatenate (
		(	
			np.random.randn(3, 2) * 0.2 + [2.5, 0.5],
			)
		)

	inputs = np.concatenate(
		( classA , classB )
	)
	targets = np.concatenate (
		(np.ones(classA.shape[0]),
		 -np.ones(classB.shape[0])))
	return inputs, targets

def TestResults(inputs,targets,results,iterations):
	print("Testing...")
	print("Start classifing without kernel")
	original = LogisticRegression( max_iter=iterations).fit(inputs, targets)

	print(original.score(inputs,targets))

	print("Start classifing with kernel")
	new = LogisticRegression( max_iter=iterations).fit(V, targets)
	print(new.score(V,targets))
	print("Input features:")
	print(inputs)
	print("Post-kernel features:")
	print(V)
	print("Prediction output with original features:")
	print(original.predict(inputs))
	print("Predicion output with post-kernel features:")
	print(new.predict(V))
	print("Correct output:")
	print(targets)

def generateAffinityMatrix(inputs):
	N=inputs.shape[0]
	Affinity_Matrix = []
	for i in range(N):
		Affinity_Matrix.append(np.zeros(N))


	for i in range(N):
		for j in range(N):
			Affinity_Matrix[i][j]=radialBasisFunctionKernel(inputs[i],inputs[j])
			if(i==j):
				Affinity_Matrix[i][j]=0
	K=np.matrix(Affinity_Matrix)
	return K

def generateDMatrix(K):
	N=len(K)
	D= []
	for i in range(N):
		D.append(np.zeros(N))
		D[i][i]=np.sum(K[i])
	D=np.matrix(D)
	return D

def generateLMatrix(K,D):
	D=fractional_matrix_power(D,-0.5)
	L=D*K*D
	L=np.matrix(L)
	return L

def getEigenvectorMatrix(L,k):
	w,V=np.linalg.eig(L)
	V=V[:k]
	V=np.matrix(V)
	V=V.transpose()
	return V
def normalizeRow(V):
	for i in range(N):
		norm=np.linalg.norm(V[i])
		V[i]=V[i]/norm
	#for i in range(N): Wanted to check that the sum of the square of the row was 1
	#s=0
	#for j in range(k):
	#s=s+V[i,j]**2
	#print(s)
	return V



inputs,targets=loadDataset('irisX.txt','irisY.txt')		#load data from file
#inputs,targets=generateDataset()
#inputs,targets=load_digits(n_class=10, return_X_y=True)
inputs,targets=randomize(inputs,targets)
N=inputs.shape[0]
print(N) 
k = 3 #Desired NUMBER OF CLUSTERS (small k)
K=generateAffinityMatrix(inputs)  #(uppercase K) STEP 1 
#w,V=np.linalg.eig(K) Here i was checking that only the first k autovalues are > 1 and this is true for all the dataset i tested.
#print(w[:3])
#print(Affinity_Matrix)

D=generateDMatrix(K) #STEP 2
L=generateLMatrix(K,D)
V=getEigenvectorMatrix(L,k) #Step 3
V=normalizeRow(V) #Step 4


#print(V)
#print(targets)

TestResults(inputs,targets,V,1000) #Test the results

#Output the new points representation (does only works with 3D points)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(V.transpose()[0], V.transpose()[1],V.transpose()[2])
#ax.scatter(V.transpose()[0], V.transpose()[1])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
#plt.figure()
#plt.plot(inputs.transpose()[0],inputs.transpose()[1],'rs')
#plt.show()
#plt.plot(np.array(V.transpose()[0])[0],np.array(V.transpose()[1])[0],'bs')




