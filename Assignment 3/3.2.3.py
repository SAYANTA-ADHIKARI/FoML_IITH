# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import explained_variance_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy as sp
import os

data = np.array([[float(entry) for entry in line.split(',')] for line in open('Residential-Building-Data-Set.csv')])
# scaler = MinMaxScaler()
# data = scaler.fit_transform(data)

X = data[:, :-2]
Y = data[:, -2:]

train_idx, test_idx = next(ShuffleSplit(n_splits=1, test_size=0.5, random_state=33).split(X, Y))
X_train, X_test, Y_train, Y_test = X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]

outfile = open('3.2.3.txt', 'w')
outfile.write(f"Train indices: {train_idx}\n")
outfile.write(f"Test indices: {test_idx}\n\n")

# Generative Linear Regression
outfile.write("Generative Linear Regression\n")
outfile.write("For inference, Y = (X_test - mean(X_train)) @ W + b\n")
outfile.write("Also center Y_test using mean(Y_train) for computing explained variance.\n")
XY_train = np.concatenate([X_train, Y_train], axis=-1)
sigma = sum([x[:, None] @ x[None, :] for x in XY_train]) / XY_train.shape[0]
Lambda = np.linalg.inv(sigma)

X_mean = np.mean(X_train, axis=0)
Y_mean = np.mean(Y_train, axis=0)

Xc = X_train - X_mean[None, :]
Yc = Y_train - Y_mean[None, :]
Xc_test = X_test - X_mean[None, :]
Yc_test = Y_test - Y_mean[None, :]

w = (np.linalg.pinv(Xc)) @ Yc
w0 = Y_mean - (X_mean @ w)
pred = Xc_test @ w + w0[None, :]
outfile.write(f"W = {w}\n")
outfile.write(f"b = {w0}\n")
outfile.write(f"Explained variances: [{explained_variance_score(Yc_test[:, 0], pred[:, 0])}, {explained_variance_score(Yc_test[:, 1], pred[:, 1])}]\n\n")
print(explained_variance_score(Yc_test, pred))

# Linear Gaussian System
outfile.write("Linear Gaussian System\n")
outfile.write("For inference, Y = X_test @ W + b\n")
sigma1 = sum([x[:, None] @ x[None, :] for x in Y_train]) / Y_train.shape[0]
sigma1_sqrt = sp.linalg.sqrtm(sigma1)
sigma1_inv_sqrt = np.linalg.inv(sigma1_sqrt)
Y_cap_train = sigma1_inv_sqrt @ Y_train.T
Y_cap_train = Y_cap_train.T
Y_cap_test = sigma1_inv_sqrt @ Y_train.T
Y_cap_test = Y_cap_test.T
model3 = LinearRegression().fit(X_train, Y_cap_train)
W_cap = sigma1_sqrt @ model3.coef_
b_cap = sigma1_sqrt @ model3.intercept_
y_pred = X_test @ W_cap.T + b_cap
outfile.write(f"W = {W_cap.T}\n")
outfile.write(f"b = {b_cap}\n")
outfile.write(f"Explained variances: [{explained_variance_score(Y_test[:, 0], y_pred[:, 0])}, {explained_variance_score(Y_test[:, 1], y_pred[:, 1])}]\n\n")
print(explained_variance_score(Y_test, y_pred))

# Linear Regression
outfile.write("Scikit-Learn Linear Regression\n")
outfile.write("For inference, Y = X_test @ W + b\n")
model = LinearRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_test)
outfile.write(f"W = {model.coef_.T}\n")
outfile.write(f"b = {model.intercept_}\n")
outfile.write(f"Explained variances: [{explained_variance_score(Y_test[:, 0], pred[:, 0])}, {explained_variance_score(Y_test[:, 1], pred[:, 1])}]\n\n")
print(explained_variance_score(Y_test, pred))
outfile.close()
# %%
