from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

# Dataset 2
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
X2 = cdc_diabetes_health_indicators.data.features
y2 = cdc_diabetes_health_indicators.data.targets  

X2= X2.values
y2 = y2.values
y2 = y2.reshape(-1)

# Scale features
scaler = StandardScaler()
X2_scaled = scaler.fit_transform(X2)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2_scaled, y2, test_size=0.6, random_state=42)

class LogisticRegressionModel:
    def __init__(self, add_bias=True, learning_rate=0.01, num_iterations=2000, batch_size=64, tolerance=1e-5):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.coefficients = None
        self.intercept = None
        self.tolerance = tolerance
        self.add_bias = add_bias

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def gradient(self, x, y):
        N,D = x.shape
        yh = self.sigmoid(np.dot(x, self.coefficients))    
        grad = np.dot(x.T, (yh - y))/N        
        return grad    

    def fit(self, x, y):
        N, D = x.shape
        # rng = np.random.default_rng()
        # batch = rng.choice(np.arange(N), self.batch_size, replace=False)
        # X_batch = x[batch, :]
        # y_batch = y[batch]
        # if self.add_bias:
        #     N = X_batch.shape[0]
        #     X = np.column_stack([X_batch,np.ones(N)])
        # N,D = X.shape
        # self.coefficients = np.zeros(D)
        
        g = np.inf 
        t = 0
        # the code snippet below is for gradient descent
        while np.linalg.norm(g) > self.tolerance and t < self.num_iterations:
            rng = np.random.default_rng()
            batch = rng.choice(np.arange(N), self.batch_size, replace=False)
            X_batch = x[batch, :]
            y_batch = y[batch]
            if self.add_bias:
                N = X_batch.shape[0]
                X = np.column_stack([X_batch,np.ones(N)])
            N,D = X.shape
            self.coefficients = np.zeros(D)
            g = self.gradient(X, y_batch)
            self.coefficients = self.coefficients - self.learning_rate * g 
            t += 1


    # def fit(self, X, y):
        
    #     m, n = X.shape
    #     self.coefficients = np.zeros(n)
    #     self.intercept = 0

    #     for _ in range(self.num_iterations):
    #         for i in range(0, m, self.batch_size):
    #             X_batch = X[i:i+self.batch_size]
    #             y_batch = y[i:i+self.batch_size]
    #             N,D = X_batch.shape

    #             z = X_batch.dot(self.coefficients) + self.intercept
    #             h = self.sigmoid(z)
    #             gradient = X_batch.T.dot(h - y_batch) / N

    #             self.coefficients -= self.learning_rate * gradient
    #             self.intercept -= self.learning_rate * np.mean(h - y_batch)

    def predict(self, x):
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        yh = self.sigmoid(np.dot(x,self.coefficients))   
        print(yh)       
        return (yh >= 0.5).astype(int)

def cost_fn(x, y, w):

        N, D = x.shape
        x = np.column_stack([x,np.ones(N)])
        z = np.dot(x, w)
        J = np.mean(y * np.log1p(np.exp(-z)) + (1-y) * np.log1p(np.exp(z)))  #log1p calculates log(1+x) to remove floating point inaccuracies
        return J
    
    
model = LogisticRegressionModel()
model.fit(X2_train, y2_train)

# plt.plot(range(len(y_test_pred_2)), y2_test, 'o')
# plt.plot(range(len(y_test_pred_2)), y_test_pred_2, 'o')
# plt.show()

y2_train_pred = model.predict(X2_train)
y2_test_pred = model.predict(X2_test)

train_accuracy = accuracy_score(y2_train, y2_train_pred)
test_accuracy = accuracy_score(y2_test, y2_test_pred)
train_precision = precision_score(y2_train, y2_train_pred)
test_precision = precision_score(y2_test, y2_test_pred)
train_recall = recall_score(y2_train, y2_train_pred)
test_recall = recall_score(y2_test, y2_test_pred)
train_f1 = f1_score(y2_train, y2_train_pred)
test_f1 = f1_score(y2_test, y2_test_pred)

print()
print("Logistic Regression Performance")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Training Precision: {train_precision:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Training Recall: {train_recall:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Training F-1 score: {train_f1:.4f}")
print(f"Test F-1 score: {test_f1:.4f}")

