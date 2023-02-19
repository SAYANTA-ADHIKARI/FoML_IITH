import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import normalize
from scipy.io import loadmat

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 7)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

data = loadmat('DrivFace/DrivFace.mat')
input_data = data['drivFaceD'][0][0][0]
input_data = normalize(input_data, norm='l2', axis=0)

out = pd.read_csv('DrivFace/drivPoints.txt')
# output_cols = ['xF', 'yF', 'wF', 'hF']
output_cols = ['xF']
output_data = out[output_cols]
output_data = normalize(output_data, norm='l2', axis=0)

X_train, X_test, Y_train, Y_test = train_test_split(input_data, output_data, test_size=0.2, random_state = 33)

# Ridge Regression with Regularization and Cross Validation
kf = KFold(n_splits=3, shuffle=True)
C_given = [0.01, 0.1, 1, 10, 100]
acc = []
for c in C_given:
    # print(c)
    acc_i = 0
    for train_idx, test_idx in kf.split(X_train):
        train_X = X_train[train_idx, :]; train_Y = Y_train[train_idx]
        test_X = X_train[test_idx, :]; test_y = Y_train[test_idx]
        ridge = Ridge(alpha=c)
        ridge.fit(train_X, train_Y)
        acc_i = acc_i + explained_variance_score(test_y, ridge.predict(test_X))/3
    # print(acc_i)
    acc.append(acc_i)

# print(acc)
plt.plot(acc, 'bo-')
plt.title("Validation Accuracy vs C")
plt.xlabel("C (= [0.01, 0.1, 1, 10, 100]) indexes --> ")
plt.xticks([0, 1, 2, 3, 4])
plt.ylabel("Validation Accuracy")
plt.savefig("Ridge_Cross.jpg", )
plt.show()

C_best = C_given[acc.index(max(acc))]

ridge = Ridge(alpha = C_best)
ridge.fit(X_train, Y_train)
print("Ridge Regression with C = ", C_best, "and Test set Accuracy as: ",  explained_variance_score(Y_test, ridge.predict(X_test)))


# Support Vector Regression with Regularization and Cross Validation
kf = KFold(n_splits=3, shuffle=True)
C_given = [0.01, 0.1, 1, 10, 100]
acc = []
for c in C_given:
    # print(c)
    acc_i = 0
    for train_idx, test_idx in kf.split(X_train):
        train_X = X_train[train_idx, :]; train_Y = Y_train[train_idx]
        test_X = X_train[test_idx, :]; test_y = Y_train[test_idx]
        svr = LinearSVR(C=c, loss='squared_epsilon_insensitive')
        svr.fit(train_X, train_Y.ravel())
        acc_i = acc_i + explained_variance_score(test_y, svr.predict(test_X))/3
    # print(acc_i)
    acc.append(acc_i)

# print(acc)
plt.plot(acc, 'bo-')
plt.title("Validation Accuracy vs C")
plt.xlabel("C (= [0.01, 0.1, 1, 10, 100]) indexes --> ")
plt.xticks([0, 1, 2, 3, 4])
plt.ylabel("Validation Accuracy")
plt.savefig("SVR_Cross.jpg", )
plt.show()

C_best = C_given[acc.index(max(acc))]

svr = LinearSVR(C=C_best, loss='squared_epsilon_insensitive')
svr.fit(X_train, Y_train.ravel())
print("Support Vector Regression with C = ", C_best, "and Test set Accuracy as: ",  explained_variance_score(Y_test, svr.predict(X_test)))

