# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from scipy.linalg import sqrtm


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 7)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
# %%
# Data Reading
data = pd.read_excel('Residential-Building-Data-Set.xlsx', sheet_name = 0, header = 1)
X = data.iloc[:, 4:-2]
Y = data.iloc[:, -2:]
print(X.head())
print(Y.head())

# %%
# Train Test split
x_train, x_test, y_train, y_test = train_test_split(X.to_numpy(), Y.to_numpy(), test_size = 0.5, random_state = 433)
print(x_train.shape)
print(y_test.shape)
# %%
# Concatenated training set
z_train = np.concatenate((x_train, y_train), axis = 1)
print(z_train.shape)

z_mean = (np.sum(z_train, axis=0) * (1/z_train.shape[0])).reshape(-1, 1)

z_cov = np.zeros((z_train.shape[1], z_train.shape[1]))
for i in range(z_train.shape[0]):
    z_cov = z_cov + ((z_train[i, :] - z_mean).T @ (z_train[i, :] - z_mean)) * (1/z_train.shape[0])
print(z_cov.shape)

z_pre = np.linalg.inv(z_cov)
print(z_pre.shape)

# %%
# First getting the w and b
x_hat = x_train @ np.linalg.inv(sqrtm(z_cov[ :103 , :103]).real)

lr1 = LinearRegression()
lr1.fit(y_train, x_hat)

w = (sqrtm(z_cov[ :103 , :103]).real) @ lr1.coef_

b = ((sqrtm(z_cov[ :103 , :103]).real) @ lr1.intercept_).reshape(-1, 1)

# %%
# Linear Gaussian System
def predict_lgs(x):
    var = np. linalg.inv(w.T @ z_pre[:103, :103] @ w + z_pre[-2: ,-2:])
    return var @ (w.T @ z_pre[:103, :103] @ (x - b) + z_pre[-2: ,-2:] @ z_mean[-2:, :])

y_pred_lgs = []
for i in range(x_test.shape[0]):
    y_pred_lgs.append(predict_lgs(x_test[i].reshape(-1, 1)).reshape(-1, ))

y_pred_lgs = np.array(y_pred_lgs)
# %%
print(explained_variance_score(y_test, y_pred_lgs))
# %%
lr2 = LinearRegression()
lr2.fit(x_train, y_train)
explained_variance_score(y_test, lr2.predict(x_test))

# %%
def predict_glr(x):
    return z_mean[-2:] + z_cov[-2:, :103] @ z_pre[:103, :103] @ (x - z_mean[:103])

y_pred_glr = []
for i in range(x_test.shape[0]):
    y_pred_glr.append(predict_glr(x_test[i].reshape(-1, 1)).reshape(-1, ))

y_pred_glr = np.array(y_pred_glr)

print(explained_variance_score(y_test, y_pred_glr))
# %%
