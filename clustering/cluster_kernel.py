import random , math
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

def radialBasisFunctionKernel(x,y):
	sigma=4
	X=np.array(x)
	Y=np.array(y)
	return np.exp(- np.linalg.norm(X-Y)/(2*sigma**2))

def loadDataset(fileX, fileY):
	x = np.genfromtxt(fileX, delimiter=',')
	y = np.genfromtxt(fileY, delimiter=',',dtype=np.int)-1
	y=np.array([k if k == 1 else -1 for k in y])
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
		(	np.random.randn(20, 2) * 0.2 + [-1.5, 0.5],
			np.random.randn(20, 2) * 0.2 + [.75, -1.5])
	)

	classB = np. concatenate (
		(	
			np.random.randn(20, 2) * 0.2 + [2.5, 0.5],
			)
		)

	inputs = np.concatenate(
		( classA , classB )
	)
	targets = np.concatenate (
		(np.ones(classA.shape[0]),
		 -np.ones(classB.shape[0])))
	return inputs, targets


inputs,targets=loadDataset('irisX_small.txt','irisY_small.txt')		#load data from file
#inputs,targets=generateDataset()
inputs,targets=randomize(inputs,targets)
N=inputs.shape[0]
print(N)
k = 2 #NUMBER OF CLUSTERS

Affinity_Matrix = []
for i in range(N):
	Affinity_Matrix.append(np.zeros(N))


for i in range(N):
	for j in range(N):
		Affinity_Matrix[i][j]=radialBasisFunctionKernel(inputs[i],inputs[j])
		if(i==j):
			Affinity_Matrix[i][j]=0

K=np.matrix(Affinity_Matrix)
w,V=np.linalg.eig(K)
#print(np.sum(w))

#print(Affinity_Matrix)


D= []
for i in range(N):
	D.append(np.zeros(N))
	D[i][i]=np.sum(K[i])
D=np.matrix(D)
#print(D)

#D=np.linalg.matrix_power(D,-1/2)
D=D**-1/2
L=D*K*D
L=np.matrix(L)

w,V=np.linalg.eig(L)
V=V[:k]
V=np.matrix(V)
#print(V)
V=V.transpose()
#print(V)

fi = []

for i in range(N):
	norm=np.linalg.norm(V[i])
	V[i]=V[i]/norm
#print(V)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.scatter(V.transpose()[0], V.transpose()[1],V.transpose()[2])
ax.scatter(V.transpose()[0], V.transpose()[1])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.figure()
plt.plot(inputs.transpose()[0],inputs.transpose()[1],'rs')
plt.show()
#plt.plot(np.array(V.transpose()[0])[0],np.array(V.transpose()[1])[0],'bs')





#print(L.shape[0])
#file = open("out.txt","w")
#for i in range(N):
#	for j in range(N):
#		file.write(str(Affinity_Matrix[i][j]))
#		if(j!=N-1):
#			file.write(", ")
#	file.write("\n")

