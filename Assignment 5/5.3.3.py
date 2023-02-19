import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 7)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

data = pd.read_csv('iris.data', header=None)
out_list = data.iloc[:, 4].unique()
data = data.replace(out_list, [1, 2, 3])
input_data = data.iloc[:, :2].to_numpy()
output_data = data.iloc[:, 4]
# print(input_data[:5, :])
# print(output_data.head())

# Helper Functions
def make_meshgrid(x, y, h = 0.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, classifier, xx, yy, **params):
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

deg_given = [2, 3]
gamma_given = [0.01, 0.1, 1, 10, 100]
C_given = [0.01, 0.1, 1, 10, 100]

for c in C_given:
    model = SVC(kernel='linear', C=c)
    classifier = model.fit(input_data, output_data)
    fig, ax = plt.subplots()
    title = ('Decision Boundary for Linear Kernel with C = ' + str(c))
    X0, X1 = input_data[:, 0], input_data[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, classifier, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=output_data, cmap=plt.cm.coolwarm, s=25, edgecolors='k')
    ax.set_ylabel('Sepal width')
    ax.set_xlabel('Sepal Length')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    plt.savefig('Graphs/Kernel_Graphs/Linear_' + str(c) + '.jpg')
    # plt.show()
    
    fig, ax = plt.subplots()
    for deg in deg_given:
        model = SVC(kernel='poly', degree=deg, C=c)
        classifier = model.fit(input_data, output_data)
        title = ('Decision Boundary for Poly Kernel with C = ' + str(c) + ' and degree = ' + str(deg))
        X0, X1 = input_data[:, 0], input_data[:, 1]
        xx, yy = make_meshgrid(X0, X1)

        plot_contours(ax, classifier, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=output_data, cmap=plt.cm.coolwarm, s=25, edgecolors='k')
        ax.set_ylabel('Sepal width')
        ax.set_xlabel('Sepal Length')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        plt.savefig('Graphs/Kernel_Graphs/Poly_' + str(deg) + '_' + str(c) + '.jpg')
        # plt.show()

    fig, ax = plt.subplots()
    for gamma in gamma_given:
        model = SVC(kernel='rbf', gamma=gamma, C=c)
        classifier = model.fit(input_data, output_data)
        title = ('Decision Boundary for RBF Kernel with C = ' + str(c) + ' and gamma = ' + str(gamma))
        X0, X1 = input_data[:, 0], input_data[:, 1]
        xx, yy = make_meshgrid(X0, X1)

        plot_contours(ax, classifier, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=output_data, cmap=plt.cm.coolwarm, s=25, edgecolors='k')
        ax.set_ylabel('Sepal width')
        ax.set_xlabel('Sepal Length')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        plt.savefig('Graphs/Kernel_Graphs/RBF_G_' + str(gamma) + '_C_' + str(c) + '.jpg')
        # plt.show()

    