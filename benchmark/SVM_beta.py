import numpy , random , math
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time


def linearKernel(x,y):
	return numpy.dot(x,y)

def polinomialKernel(x,y):
	p=2
	return (numpy.dot(x,y)+1)**p

def radialBasisFunctionKernel(x,y):
	sigma=4
	X=numpy.array(x)
	Y=np.array(y)
	return np.exp(- np.linalg.norm(X-Y)/(2*sigma**2))

def preCompute():
	P=[]
	for i in range(N):
		l=numpy.zeros(N)
		for j in range(N):
			l[j]=targets[i]*targets[j]*kernel(inputs[i],inputs[j])
		P.append(l)
	return P
	
def objective(alpha):
	result = 0;
	for i in range(N):
		for j in range(N):
			result += alpha[i]*alpha[j]*P[i][j] 
	return result/2 - numpy.sum(alpha)


def zerofun(alpha):
	return numpy.dot(alpha,targets);

def indicator(s):
	result=0;
	for i in range(N):
		result+=alpha[i]*targets[i]*kernel(s,inputs[i])
	return result-b


def generateDataset():
	numpy.random.seed(100)
	classA = numpy. concatenate (
		(	numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5],
			numpy.random.randn(10, 2) * 0.2 + [.75, -1.5])
	)

	classB = numpy. concatenate (
		(	numpy.random.randn(10 , 2) * 0.2 + [-1.5 , -0.5],
			numpy.random.randn(10, 2) * 0.2 + [2.5, 0.5],
			numpy.random.randn(10, 2) * 0.2 + [0., 2.5])
		)

	inputs = numpy.concatenate(
		( classA , classB )
	)
	targets = numpy.concatenate (
		(numpy.ones(classA.shape[0]),
		 -numpy.ones(classB.shape[0])))
	return inputs, targets

def loadDataset(fileX, fileY):
	x = np.genfromtxt(fileX, delimiter=',')
	y = np.genfromtxt(fileY, delimiter=',',dtype=np.int)-1
	y=np.array([k if k == 1 else -1 for k in y])
	return x,y

def partition(X,Y, fraction):
	breakPoint = int(len(X) * fraction)
	return X[:breakPoint], X[breakPoint:], Y[:breakPoint], Y[breakPoint:]


def randomize(inputs, targets):
	N = inputs.shape[0] # Number  of  rows  ( s a m p l e s )
	permute=list(range(N))
	random.shuffle(permute)
	inputs_out = inputs [permute, : ]
	targets_out = targets [ permute ]
	return inputs_out, targets_out

def train():
	C=10; #upper bound for alpha 
	B=[(0, C) for b in range(N)]
	XC={'type':'eq', 'fun':zerofun}
	start=numpy.zeros(N)
	timerStart = time.time()
	ret = minimize (objective, start, bounds=B, constraints=XC )
	alpha = ret['x']
	end = time.time()
	trueAlpha = []

	for num in alpha: #put to 0 alphas close to 0
		if(num>10e-5):
			trueAlpha.append(num);
		else:
			trueAlpha.append(0)


	for idx in range(N): #get the index of a SV
		if(trueAlpha[idx]!=0):
			break;

	b=0;
	for i in range(N): #compute b
		b+=trueAlpha[i]*targets[i]*kernel(inputs[idx],inputs[i])
	b-=targets[idx]
	#print(b)
	print(end - timerStart)
	return alpha,b


	#print(targets)
	#a=[]
	#for i in range(N):
	#	a.append(indicator(inputs[i]))
	#print(a)

	#print(ret['success'])
	


inputs,targets=loadDataset('irisX.txt','irisY.txt')		#load data from file
inputs,targets=randomize(inputs,targets)				#randomize
inputs, test_x, targets, test_y = partition(inputs,targets,0.7) # split train_set and control test
N=inputs.shape[0]
#inputs,targets=generateDataset()
#print(inputs)
#print(targets)
#plt.plot([p[0] for p in classA],  [p[1] for p in classA], 'b.')
#plt.plot([p[0] for p in classB],  [p[1] for p in classB],  'r.')

kernels = [linearKernel,polinomialKernel,radialBasisFunctionKernel] #list of kernels
for kernel in kernels:
	P=preCompute()
	alpha,b = train()

	print("Start classification of tests:")
	good=0
	for i in range(len(test_x)):
		if(np.sign(indicator(test_x[i]))==test_y[i]):
			good+=1

	print("Classificaton accuracy: "+str(100*good/len(test_x))+"%")


#	plt.axis('equal') #Forcesamescaleonbothaxes
#	#plt.savefig('svmplot.pdf')#Saveacopyinafile
#
#
#	xgrid=numpy.linspace(-5,5)
#	ygrid=numpy.linspace(-4,4)
#	grid=numpy.array([[indicator(x,y)
#					for x in xgrid]
#					for y in ygrid])
#	plt.contour(xgrid,ygrid,grid,
#	(-1.0,0.0,1.0),
#	colors=('red','black','blue'),
#	linewidths=(1,3,1))
#	plt.show()#Showtheplotonthescreen
