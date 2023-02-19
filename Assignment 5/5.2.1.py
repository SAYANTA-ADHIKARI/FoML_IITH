import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 7)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

input_data = pd.read_excel("LSVT_voice_rehabilitation.xlsx", sheet_name=0, header=0)
output_data = pd.read_excel("LSVT_voice_rehabilitation.xlsx", sheet_name=1, header=0)
# print(input_data.head())
# print(output_data.head())
input_data = normalize(input_data, norm='l2', axis=0)
output_data = output_data.to_numpy().ravel()

X_train, X_test, Y_train, Y_test = train_test_split(input_data, output_data, test_size=0.2, random_state = 33)

# Logistic Regression with Regularization and Cross Validation
kf = KFold(n_splits=3, shuffle=True)
C_given = [0.01, 0.1, 1, 10, 100]
acc = []
for c in C_given:
    # print(c)
    acc_i = 0
    for train_idx, test_idx in kf.split(X_train):
        train_X = X_train[train_idx, :]; train_Y = Y_train[train_idx]
        test_X = X_train[test_idx, :]; test_y = Y_train[test_idx]
        logreg = LogisticRegression(penalty='l2', C=c)
        logreg.fit(train_X, train_Y)
        acc_i = acc_i + accuracy_score(test_y, logreg.predict(test_X))/3
    # print(acc_i)
    acc.append(acc_i)

# print(acc)
plt.plot(acc, 'bo-')
plt.title("Validation Accuracy vs C")
plt.xlabel("C (= [0.01, 0.1, 1, 10, 100]) indexes --> ")
plt.xticks([0, 1, 2, 3, 4])
plt.ylabel("Validation Accuracy")
plt.savefig("LogReg_Cross.jpg", )
plt.show()

C_best = C_given[acc.index(max(acc))]

logreg = LogisticRegression(penalty='l2', C=C_best)
logreg.fit(X_train, Y_train)
print("Logistic Regression with C = ", C_best, "and Test set Accuracy as: ",  accuracy_score(Y_test, logreg.predict(X_test)))

# Support Vector Classifier with Regularization and Cross Validation
# X_train, X_test, Y_train, Y_test = train_test_split(input_data, output_data, test_size=0.2)

kf = KFold(n_splits=3, shuffle=True)
C_given = [0.01, 0.1, 1, 10, 100]
acc = []
for c in C_given:
    # print(c)
    acc_i = 0
    for train_idx, test_idx in kf.split(X_train):
        train_X = X_train[train_idx, :]; train_Y = Y_train[train_idx]
        test_X = X_train[test_idx, :]; test_y = Y_train[test_idx]
        svc = LinearSVC(penalty='l2', C=c)
        svc.fit(train_X, train_Y)
        acc_i = acc_i + accuracy_score(test_y, svc.predict(test_X))/3
    # print(acc_i)
    acc.append(acc_i)

# print(acc)
plt.plot(acc, 'bo-')
plt.title("Validation Accuracy vs C")
plt.xlabel("C (= [0.01, 0.1, 1, 10, 100]) indexes --> ")
plt.xticks([0, 1, 2, 3, 4])
plt.ylabel("Validation Accuracy")
plt.savefig("SVC_Cross.jpg", )
plt.show()

C_best = C_given[acc.index(max(acc))]

svc = LinearSVC(penalty='l2', C=C_best)
svc.fit(X_train, Y_train)
print("Support Vector Classifier with C = ", C_best, "and Test set Accuracy as: ",  accuracy_score(Y_test, svc.predict(X_test)))


    


