# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression, Perceptron, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, accuracy_score
from sklearn.datasets import make_spd_matrix
np.random.seed(38)

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 7)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# %%
input_data = loadmat('DrivFace/DrivFace.mat')
X = input_data['drivFaceD'][0][0][0]
# print (input_data)
# print(X)
# print(X.shape)

output_data = pd.read_csv('DrivFace/drivPoints.txt')
# print (output_data.columns)
output_cols = ['xF', 'yF', 'wF', 'hF']
Y = output_data[output_cols]
# print(Y)
# print(Y.shape)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.5)
# print (Xtrain.shape)
# print (Ytest.shape)

model1 = LinearRegression()
model1.fit(Xtrain, Ytrain)
Ypred = model1.predict(Xtest)
metric1 = explained_variance_score(Ytest, Ypred)
print('Explained Variance with 1st feature map: ', metric1)

X_new = np.concatenate([np.ones((606,1)), X, X**2], axis = 1)
# print(X_new.shape)

X_newtrain, X_newtest, Ytrain, Ytest = train_test_split(X_new, Y, test_size = 0.5)
# print (Xtrain.shape)
# print (Ytest.shape)

model2 = LinearRegression()
model2.fit(X_newtrain, Ytrain)
Y_newpred = model2.predict(X_newtest)
metric2 = explained_variance_score(Ytest, Y_newpred)
print('Explained Variance with 2nd feature map: ', metric2)

# %%
m = 500
max_d = 800
ex_var = []
for d in range(1, max_d+1):
    mean = [0 for _ in range(d)]
    X = np.random.multivariate_normal(mean, np.eye(d), size = 2*m)
    W = np.random.randn(d, 1)
    Y = np.random.randn(2*m, 1) + X @ W
    # print(X.shape)
    # print(W.shape)
    # print(Y.shape)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.5)
    model = LinearRegression()
    model.fit(Xtrain, Ytrain)
    Ypred = model.predict(Xtest)
    ex_var.append(explained_variance_score(Ytest, Ypred))

plt.plot(range(1, max_d+1), ex_var)
plt.title('Explained Varience on Test set vs. d')
plt.xlabel('d values')
plt.ylabel('Explained Variance on Test set')


# %%
input_data = pd.read_excel('LSVT_voice_rehabilitation.xlsx')
output_data = pd.read_excel('LSVT_voice_rehabilitation.xlsx', sheet_name = 1)
# print(input_data.shape)
# print(output_data.shape)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(input_data, output_data, test_size = 0.5, random_state = 33)
# print(Xtrain.shape)
# print(Ytest.shape)

model1 = Perceptron()
model1.fit(Xtrain, Ytrain)
Ypred = model1.predict(Xtrain)
print('Accuracy Score on Perceptron with 1st feature map: ', accuracy_score(Ytest, Ypred))

model2 = LogisticRegression(penalty='none')
model2.fit(Xtrain, Ytrain)
Ypred = model2.predict(Xtrain)
print('Accuracy Score on Logistic Regression with 1st feature map: ',accuracy_score(Ytest, Ypred))

new_input_data = np.concatenate([np.ones((126, 1)), input_data, input_data**2], axis = 1)
# print(new_input_data.shape)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(new_input_data, output_data, test_size = 0.5, random_state = 38)
# print(Xtrain.shape)
# print(Ytest.shape)

model3 = Perceptron()
model3.fit(Xtrain, Ytrain)
Ypred = model3.predict(Xtrain)
print('Accuracy Score on Perceptron with 2nd feature map: ',accuracy_score(Ytest, Ypred))

model4 = LogisticRegression(penalty='none')
model4.fit(Xtrain, Ytrain)
Ypred = model4.predict(Xtrain)
print('Accuracy Score on Logistic Regression with 2nd feature map: ',accuracy_score(Ytest, Ypred))

# %%
A = make_spd_matrix(10)
b = np.random.rand(10,1)
c = np.random.rand()

def objective(v):
    obj = np.transpose(v) @ A @ v - 2 * np.dot(np.transpose(b), v) + c
    return obj.ravel()

# Analitically solving the equation
v_ana = np.linalg.solve(A, b)
obj_ana = objective(v_ana)
# print(v_ana)
# print(obj_ana)

# Gradient Descent
v_GD = np.array([0.1 for _ in range(10)]).reshape(-1, 1)
step_size_GD = 1/(2 * np.linalg.norm(A, ord = 2) + np.linalg.norm(b))

obj_itr_GD = []
obj_itr_GD.append(objective(v_GD))
for _ in range(1000):
    v_GD = v_GD - step_size_GD * (2 * A @ v_GD - 2 * b)
    obj_itr_GD.append(objective(v_GD))

# print(v_GD)
# print(objective(v_GD))

# Stochastic Gradient Descent
v_SGD = np.array([0.1 for _ in range(10)]).reshape(-1, 1)
step_size_SGD = step_size_GD/100

obj_itr_SGD = []
obj_itr_SGD.append(objective(v_SGD))
for _ in range(100000):
    v_SGD = v_SGD - step_size_SGD * ((2 * A @ v_SGD - 2 * b) + 0.5 * np.random.randn(10, 1))
    obj_itr_SGD.append(objective(v_SGD))

# print(v_SGD)
# print(objective(v_SGD).shape)
ana_list = [obj_ana for _ in range(100000)]

plt.figure()
plt.plot(ana_list[:1000], label = 'Analitical objective Value')
plt.plot(obj_itr_GD, label = 'Objective in Gradient Descent')
plt.legend()
plt.title('Objective Function value vs. iterations')
plt.xlabel('Iterations')
plt.ylabel('Objective funtion value')

plt.figure()
plt.plot(ana_list, label = 'Analitical objective Value')
plt.plot(obj_itr_SGD, label = 'Objective in Stochastic Gradient Descent')
plt.legend()
plt.title('Objective Function value vs. iterations')
plt.xlabel('Iterations')
plt.ylabel('Objective funtion value')

plt.figure()
plt.plot(ana_list, label = 'Analytical objective Value')
plt.plot(obj_itr_GD, label = 'Objective in Gradient Descent')
plt.plot(obj_itr_SGD, label = 'Objective in Stochastic Gradient Descent')
plt.legend()
plt.title('Objective Function value vs. iterations')
plt.xlabel('Iterations')
plt.ylabel('Objective funtion value')

# %%
