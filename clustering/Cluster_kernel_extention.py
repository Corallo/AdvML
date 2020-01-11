import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from scipy.sparse.linalg import eigs
from sklearn.datasets import make_circles
from sklearn import svm
from digit_datasets import generateDigitsDataset
from new_dataset import get_20newsgroup_tf_idf
def radialBasisFunctionKernel(x, y, sigma=5):
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


def transfer_function(L, k, function):
    t = 3
    w, V = np.linalg.eig(L)
    w = np.diag(w)
    if function == "linear":
        return w
    if function == "step":
        w, V = np.linalg.eig(L)
        w_cut = np.sort(w)[k - 1]
        w = np.diag(w)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                if w[i, j] >= w_cut:
                    w[i, j] = 1
                else:
                    w[i, j] = 0

    if function == "linear step":
        w, V = np.linalg.eig(L)
        w_cut = np.sort(w)[k - 1]
        w = np.diag(w)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                if w[i, j] >= w_cut:
                    w[i, j] = w[i, j]
                else:
                    w[i, j] = 0

    if function == "polynomial":
        w, V = np.linalg.eig(L)
        w_cut = np.sort(w)[k - 1]
        w = np.diag(w)
        w**t
    return w


def transformL(L):
    w, U = np.linalg.eig(L)
    L_new =  U.dot(transfer_function(L,k,"polynomial")).dot(np.linalg.inv(U))
    return L_new.real

def transformDMatrix(L):
	D_new = np.zeros((L.shape))
	for i in range(D_new.shape[0]):
		D_new[i,i] = 1/L_new[i,i]
	return D_new

def transformKMatrix(D_new,L_new):
	K_new = fractional_matrix_power(D_new,0.5).dot(L_new).dot(fractional_matrix_power(D_new,0.5))
	return K_new

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




#inputs, targets = make_circles(n_samples=200, shuffle=True, noise=None, random_state=150, factor=0.4)
# inputs,targets=loadDataset('irisX_small.txt','irisY_small.txt') load data from file
# inputs,targets=generateDataset()
inputs,targets=generateDigitsDataset()
#inputs,targets=get_20newsgroup_tf_idf("all", ["comp.os.ms-windows.misc", "comp.sys.mac.hardware"], 7511)
#inputs,targets=load_digits(n_class=10, return_X_y=True)
inputs=np.array(inputs)
targets=np.array(targets)
inputs, targets = randomize(inputs, targets)
print(np.shape(inputs))
print(np.shape(targets))

N = inputs.shape[0]
#print(N)
k = 2  # Desired NUMBER OF CLUSTERS (small k)
K = generateAffinityMatrix(inputs)  # (uppercase K) STEP 1
D = generateDMatrix(K)  # STEP 2
L = generateLMatrix(K, D)
L_new = transformL(L)
D_new = transformDMatrix(L)
K_new = transformKMatrix(D_new,L_new)
# plt.plot(V[:,0],V[:,1],'rs')

#standard_prediction, prediction = TestResults(inputs, targets, K_new, 1000)  # Test the results
standard_prediction, prediction = TestWithSVM(inputs,targets,K_new)
plotOutput(inputs, standard_prediction)
plotOutput(inputs, prediction)
plt.show()
#print(np.shape(K_new))
# Output the new points representation (does only works with 3D points)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
##ax.scatter(V.transpose()[0], V.transpose()[1],V.transpose()[2])
#ax.scatter(V.transpose()[0], V.transpose()[1])
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
# plt.show()

# plt.figure()
# plt.plot(inputs.transpose()[0],inputs.transpose()[1],'rs')
# plt.show()
# plt.plot(np.array(V.transpose()[0])[0],np.array(V.transpose()[1])[0],'bs')
