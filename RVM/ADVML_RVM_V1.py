#!/usr/bin/env python
# coding: utf-8

# In[66]:


from sklearn import datasets
import skrvm #https://github.com/JamesRitchie/scikit-rvm
iris = datasets.load_iris()
digits = datasets.load_digits()
news = datasets.fetch_20newsgroups
news_train = news(subset='train', categories= ['comp.os.ms-windows.misc', 'comp.sys.mac.hardware'])


# In[67]:


from sklearn.feature_extraction.text import TfidfVectorizer
#load Windows & Mac Training Data
newsgroups_train = datasets.fetch_20newsgroups(subset='train', categories = ['comp.os.ms-windows.misc', 'comp.sys.mac.hardware'])
#Load Windows & Mac Target Data
news_train_target = news(subset='train', categories= ['comp.os.ms-windows.misc', 'comp.sys.mac.hardware']).target
#TDIF
vectorizer = TfidfVectorizer()
newsgroups_train.data = vectorizer.fit_transform(newsgroups_train.data)
#Turn to dense matrix
news_train_data = newsgroups_train.data.toarray()


# In[68]:


#Code taken from https://github.com/JamesRitchie/scikit-rvm

"""Relevance Vector Machine classes for regression and classification."""
import numpy as np

from scipy.optimize import minimize
from scipy.special import expit

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics.pairwise import (
    linear_kernel,
    rbf_kernel,
    polynomial_kernel
)
from sklearn.multiclass import OneVsOneClassifier
from sklearn.utils.validation import check_X_y


class BaseRVM(BaseEstimator):

    """Base Relevance Vector Machine class.
    Implementation of Mike Tipping's Relevance Vector Machine using the
    scikit-learn API. Add a posterior over weights method and a predict
    in subclass to use for classification or regression.
    """

    def __init__(
        self,
        kernel='rbf',
        degree=3,
        coef1=None,
        coef0=0.0,
        n_iter=3000,
        tol=1e-3,
        alpha=1e-6,
        threshold_alpha=1e9,
        beta=1.e-6,
        beta_fixed=False,
        bias_used=True,
        verbose=False
    ):
        """Copy params to object properties, no validation."""
        self.kernel = kernel
        self.degree = degree
        self.coef1 = coef1
        self.coef0 = coef0
        self.n_iter = n_iter
        self.tol = tol
        self.alpha = alpha
        self.threshold_alpha = threshold_alpha
        self.beta = beta
        self.beta_fixed = beta_fixed
        self.bias_used = bias_used
        self.verbose = verbose

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = {
            'kernel': self.kernel,
            'degree': self.degree,
            'coef1': self.coef1,
            'coef0': self.coef0,
            'n_iter': self.n_iter,
            'tol': self.tol,
            'alpha': self.alpha,
            'threshold_alpha': self.threshold_alpha,
            'beta': self.beta,
            'beta_fixed': self.beta_fixed,
            'bias_used': self.bias_used,
            'verbose': self.verbose
        }
        return params

    def set_params(self, **parameters):
        """Set parameters using kwargs."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _apply_kernel(self, x, y):
        """Apply the selected kernel function to the data."""
        if self.kernel == 'linear':
            phi = linear_kernel(x, y)
        elif self.kernel == 'rbf':
            phi = rbf_kernel(x, y, self.coef1)
        elif self.kernel == 'poly':
            phi = polynomial_kernel(x, y, self.degree, self.coef1, self.coef0)
        elif callable(self.kernel):
            phi = self.kernel(x, y)
            if len(phi.shape) != 2:
                raise ValueError(
                    "Custom kernel function did not return 2D matrix"
                )
            if phi.shape[0] != x.shape[0]:
                raise ValueError(
                    "Custom kernel function did not return matrix with rows"
                    " equal to number of data points."""
                )
        else:
            raise ValueError("Kernel selection is invalid.")

        if self.bias_used:
            phi = np.append(phi, np.ones((phi.shape[0], 1)), axis=1)

        return phi

    def _prune(self):
        """Remove basis functions based on alpha values."""
        keep_alpha = self.alpha_ < self.threshold_alpha

        if not np.any(keep_alpha):
            keep_alpha[0] = True
            if self.bias_used:
                keep_alpha[-1] = True

        if self.bias_used:
            if not keep_alpha[-1]:
                self.bias_used = False
            self.relevance_ = self.relevance_[keep_alpha[:-1]]
        else:
            self.relevance_ = self.relevance_[keep_alpha]

        self.alpha_ = self.alpha_[keep_alpha]
        self.alpha_old = self.alpha_old[keep_alpha]
        self.gamma = self.gamma[keep_alpha]
        self.phi = self.phi[:, keep_alpha]
        self.sigma_ = self.sigma_[np.ix_(keep_alpha, keep_alpha)]
        self.m_ = self.m_[keep_alpha]

    def fit(self, X, y):
        """Fit the RVR to the training data."""
        X, y = check_X_y(X, y)

        n_samples, n_features = X.shape

        self.phi = self._apply_kernel(X, X)

        n_basis_functions = self.phi.shape[1]

        self.relevance_ = X
        self.y = y

        self.alpha_ = self.alpha * np.ones(n_basis_functions)
        self.beta_ = self.beta

        self.m_ = np.zeros(n_basis_functions)

        self.alpha_old = self.alpha_

        for i in range(self.n_iter):
            self._posterior()

            self.gamma = 1 - self.alpha_*np.diag(self.sigma_)
            self.alpha_ = self.gamma/(self.m_ ** 2)

            if not self.beta_fixed:
                self.beta_ = (n_samples - np.sum(self.gamma))/(
                    np.sum((y - np.dot(self.phi, self.m_)) ** 2))

            self._prune()

            if self.verbose:
                print("Iteration: {}".format(i))
                print("Alpha: {}".format(self.alpha_))
                print("Beta: {}".format(self.beta_))
                print("Gamma: {}".format(self.gamma))
                print("m: {}".format(self.m_))
                print("Relevance Vectors: {}".format(self.relevance_.shape[0]))
                print()

            delta = np.amax(np.absolute(self.alpha_ - self.alpha_old))

            if delta < self.tol and i > 1:
                break

            self.alpha_old = self.alpha_

        if self.bias_used:
            self.bias = self.m_[-1]
        else:
            self.bias = None

        return self


class RVR(BaseRVM, RegressorMixin):

    """Relevance Vector Machine Regression.
    Implementation of Mike Tipping's Relevance Vector Machine for regression
    using the scikit-learn API.
    """

    def _posterior(self):
        """Compute the posterior distriubtion over weights."""
        i_s = np.diag(self.alpha_) + self.beta_ * np.dot(self.phi.T, self.phi)
        self.sigma_ = np.linalg.inv(i_s)
        self.m_ = self.beta_ * np.dot(self.sigma_, np.dot(self.phi.T, self.y))

    def predict(self, X, eval_MSE=False):
        """Evaluate the RVR model at x."""
        phi = self._apply_kernel(X, self.relevance_)

        y = np.dot(phi, self.m_)

        if eval_MSE:
            MSE = (1/self.beta_) + np.dot(phi, np.dot(self.sigma_, phi.T))
            return y, MSE[:, 0]
        else:
            return y


class RVC(BaseRVM, ClassifierMixin):

    """Relevance Vector Machine Classification.
    Implementation of Mike Tipping's Relevance Vector Machine for
    classification using the scikit-learn API.
    """

    def __init__(self, n_iter_posterior=50, **kwargs):
        """Copy params to object properties, no validation."""
        self.n_iter_posterior = n_iter_posterior
        super(RVC, self).__init__(**kwargs)

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = super(RVC, self).get_params(deep=deep)
        params['n_iter_posterior'] = self.n_iter_posterior
        return params

    def _classify(self, m, phi):
        return expit(np.dot(phi, m))

    def _log_posterior(self, m, alpha, phi, t):

        y = self._classify(m, phi)

        log_p = -1 * (np.sum(np.log(y[t == 1]+.00000001), 0) +
                      np.sum(np.log(1-y[t == 0]+.00000001), 0))
        log_p = log_p + 0.5*np.dot(m.T, np.dot(np.diag(alpha), m))

        jacobian = np.dot(np.diag(alpha), m) - np.dot(phi.T, (t-y))

        return log_p, jacobian

    def _hessian(self, m, alpha, phi, t):
        y = self._classify(m, phi)
        B = np.diag(y*(1-y))
        return np.diag(alpha) + np.dot(phi.T, np.dot(B, phi))

    def _posterior(self):
        result = minimize(
            fun=self._log_posterior,
            hess=self._hessian,
            x0=self.m_,
            args=(self.alpha_, self.phi, self.t),
            method='Newton-CG',
            jac=True,
            options={
                'maxiter': self.n_iter_posterior
            }
        )

        self.m_ = result.x
        self.sigma_ = np.linalg.inv(
            self._hessian(self.m_, self.alpha_, self.phi, self.t)
        )

    def fit(self, X, y):
        """Check target values and fit model."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("Need 2 or more classes.")
        elif n_classes == 2:
            self.t = np.zeros(y.shape)
            self.t[y == self.classes_[1]] = 1
            return super(RVC, self).fit(X, self.t)
        else:
            self.multi_ = None
            self.multi_ = OneVsOneClassifier(self)
            self.multi_.fit(X, y)
            return self

    def predict_proba(self, X):
        """Return an array of class probabilities."""
        phi = self._apply_kernel(X, self.relevance_)
        y = self._classify(self.m_, phi)
        return np.column_stack((1-y, y))

    def predict(self, X):
        """Return an array of classes for each input."""
        if len(self.classes_) == 2:
            y = self.predict_proba(X)
            res = np.empty(y.shape[0], dtype=self.classes_.dtype)
            res[y[:, 1] <= 0.5] = self.classes_[0]
            res[y[:, 1] >= 0.5] = self.classes_[1]
            return res
        else:
            return self.multi_.predict(X)


# In[69]:



#PLOT IF LOWER DIMENSION
# n_grid = 500
# max_x      = np.max(news_train_data,axis = 0)
# min_x      = np.min(news_train_data,axis = 0)
# X1         = np.linspace(min_x[0],max_x[0],n_grid)
# X2         = np.linspace(min_x[1],max_x[1],n_grid)
# n_grid = 500
# max_x      = np.max(news_train_data,axis = 0)
# min_x      = np.min(news_train_data,axis = 0)
# X1         = np.linspace(min_x[0],max_x[0],n_grid)
# X2         = np.linspace(min_x[1],max_x[1],n_grid)
# x1,x2      = np.meshgrid(X1,X2)
# Xgrid      = np.zeros([n_grid**2,2])
# Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
# Xgrid[:,1] = np.reshape(x2,(n_grid**2,))

# rv_grid = rvm.predict_proba(Xgrid)[:,1]
# models  = [rv_grid]
# model_names = ["RVC"]
               
               
# for model, model_name in zip(models, model_names):
#     plt.figure(figsize = (12,8))
#     plt.contourf(X1,X2,np.reshape(model,(n_grid,n_grid)),
#                        cmap="coolwarm",
#                        figsize = (10,16))
#     plt.colorbar()
#     plt.plot(news_train_data[news_train_target==0,0],news_train_data[news_train_target==0,1],"bo", markersize = 4)
#     plt.plot(news_train_data[news_train_target==1,0],news_train_data[news_train_target==1,1],"ro", markersize = 4)
#     # plot 'relevant' vectors
#     svrv = None
#     point_label = None
#     svrv = rvm.relevant_vectors_[0]
#     point_label = "relevant vecs"
#     plt.plot(svrv[:,0],svrv[:,1],'co',markersize=8,label=point_label)
#     plt.plot()
#     title = model_name
#     plt.title(title)
#     plt.xlabel("x1")
#     plt.ylabel("x2")
#     plt.legend()
#     plt.show()

