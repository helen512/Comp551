from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# fetch dataset 
infrared_thermography_temperature = fetch_ucirepo(id=925) 
# data (as pandas dataframes) 
X = infrared_thermography_temperature.data.features 
y = infrared_thermography_temperature.data.targets 

# !compare the accuracy with x_dummies and x_drop
x_dummies = pd.get_dummies(X, columns=['Age', 'Gender', 'Ethnicity'], drop_first=True)
# x_drop = X.drop(['Age', 'Gender', 'Ethnicity'], axis=1)

bool_columns = x_dummies.select_dtypes(include=['bool']).columns
for col in bool_columns:
    x_dummies[col] = x_dummies[col].astype(int)

nan_rows = x_dummies.isnull().any(axis=1)
x_dummies= x_dummies[~nan_rows]

# nan_rows = x_drop.isnull().any(axis=1)
# x_drop= x_drop[~nan_rows]
print("NaN values in x_dummies: ", x_dummies.isnull().any().any())
# x_dummies.to_csv('datadummies.csv',index =False)
# print(x_dummies) # [1020 rows x 43 columns]
y = y['aveOralM']
y= y[~nan_rows]


# !try different scalers?
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x_dummies)

# print(x_scaler)

# !test the minimum data require to train 
X_train, X_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.4, random_state=42)

class LinearRegression():
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        # print(np.linalg.inv(X.T.dot(X)))
        # print(self.coefficients)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]
        return self

    def predict(self, X):
        f = X.dot(self.coefficients) + self.intercept
        return f
    
model = LinearRegression()
model.fit(X_train, y_train) 
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# print("X_train NaNs: ", np.isnan(X_train).any())
# print("X_test NaNs: ", np.isnan(X_test).any())
# print("y_train_pred NaNs: ", np.isnan(y_train_pred).any())
# print("y_test_pred NaNs: ", np.isnan(y_test_pred).any())
# print(y_test_pred)

#Calculate metrics
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"MSE (Train): {mse_train:.4f}")
print(f"MSE (Test): {mse_test:.4f}")
print(f"R² (Train): {r2_train:.4f}")
print(f"R² (Test): {r2_test:.4f}")
print(f"MAE (Train): {mae_train:.4f}")
print(f"MAE (Test): {mae_test:.4f}")

model2 = LinearRegression()
yh = model2.fit(X_test, y_test).predict(X_test)
plt.scatter(yh, y_test)
plt.xlabel("yh")
plt.ylabel("y_test")
plt.show()

