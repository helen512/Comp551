from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


#Data set 2
# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
X = X.values
# scaler = StandardScaler()
# X= scaler.fit_transform(X)
y = cdc_diabetes_health_indicators.data.targets 
y = y.values
y = y.reshape(-1)
  
# X.to_csv('data2x.csv',index =False)
# y.to_csv('data2y.csv', index = False)

# print("X NaNs: ", np.isnan(X).any())

def logistic(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, add_bias=True, learning_rate=.01, epsilon=1e-2, max_iters=1e5, verbose=False):
        self.add_bias = add_bias
        self.learning_rate = learning_rate
        self.epsilon = epsilon                        #to get the tolerance for the norm of gradients 
        self.max_iters = max_iters                    #maximum number of iteration of gradient descent
        self.verbose = verbose

    def gradient(self, x, y):
        N,D = x.shape
        yh = logistic(np.dot(x, self.w))    # predictions  size N
        print(y)
        print(yh)
        # print(x.T.shape, yh.shape, y.shape)
        # print(y)
        # print(yh)
        grad = np.dot(x.T, (yh - y))/N        # divide by N because cost is mean over N points
        return grad    

    def fit(self, x, y):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        N,D = x.shape
        self.w = np.zeros(D)
        
        g = np.inf 
        t = 0
        # the code snippet below is for gradient descent
        while np.linalg.norm(g) > self.epsilon and t < self.max_iters:
            g = self.gradient(x, y)
            self.w = self.w - self.learning_rate * g 
            t += 1
            print(t)
        
        if self.verbose:
            print(f'terminated after {t} iterations, with norm of the gradient equal to {np.linalg.norm(g)}')
            print(f'the weight found: {self.w}')
        return self
    
    def predict(self, x):
        if x.ndim == 1:
            x = x[:, None]
        Nt = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(Nt)])
        yh = logistic(np.dot(x,self.w))            #predict output
        return yh

def cost_fn(x, y, w):
        N, D = x.shape
        x = np.column_stack([x,np.ones(N)])           
        print(x.shape)
        print(w.shape)                                   
        z = np.dot(x, w)
        print(z.shape)
        J = np.mean(y * np.log1p(np.exp(-z)) + (1-y) * np.log1p(np.exp(z)))  #log1p calculates log(1+x) to remove floating point inaccuracies 
        return J


N = 50
x = np.linspace(-5,5, N)
y = ( x < 2).astype(int) 
model_test = LogisticRegression()
model_test.fit(x, y)

# model = LogisticRegression()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
# # print(X_train)
# # print(y_train)
# model.fit(X_train, y_train)
# print(model.predict(X_train))
# print(y_train)
# print(cost_fn(X_train, y_train, model.w))
# print(cost_fn(X_test, y_test, model.w))

# y_train_pred = model.predict(X_train)
# y_test_pred = model.predict(X_test)