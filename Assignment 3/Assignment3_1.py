# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.preprocessing import OneHotEncoder

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 7)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# %%
# Data Extraction
data = pd.DataFrame()
for Folder in os.listdir(path= os.path.join(os.getcwd(), 'Parkinson_hw')):
    if Folder == 'hw_drawings' or Folder == 'readme.txt':
        pass
    else:
        # print(Folder)
        for folder in os.listdir(path= os.path.join(os.getcwd(), 'Parkinson_hw', Folder)):
            for file in os.listdir(path= os.path.join(os.getcwd(), 'Parkinson_hw', Folder, folder)):
                temp = pd.read_csv(os.path.join(os.getcwd(), 'Parkinson_hw', Folder, folder, file), sep=';', header=None)
                data = pd.concat([data, temp])

print(data.shape)
# %%
# Input Output segregation
Y = data.iloc[:, 6]

X = data.iloc[:, 3:5]
# Normalizing X
scaler = StandardScaler()

X = []
for i in Y.unique():
    X.append(scaler.fit_transform(data[Y==i].iloc[:, 3:5]))


# %%
# Functions required
def mvn2d(x, y, u, sigma):
    xx, yy = np.meshgrid(x, y)
    xy = np.c_[xx.ravel(), yy.ravel()]
    sigma_inv = np.linalg.inv(sigma)
    z = np.dot((xy - u), sigma_inv)
    z = np.sum(z * (xy - u), axis=1)
    z = np.exp(-0.5 * z)
    z = z / (2 * np.pi * np.linalg.det(sigma) ** 0.5)
    return z.reshape(xx.shape) 

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def make_grid(x):
    points = np.vstack(x)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_range = np.linspace(x_min - 1, x_max + 1, ngrid)
    y_range = np.linspace(y_min - 1, y_max + 1, ngrid)
    xx, yy = np.meshgrid(x_range, y_range)
    return xx, yy, x_range, y_range

def plot_dboundaries(xx, yy, z, z_p):
    plt.pcolormesh(xx, yy, z, alpha=0.1)
    plt.jet()
    nclasses = z_p.shape[1]
    for j in range(nclasses):
        plt.contour(xx, yy, z_p[:, j].reshape(ngrid, ngrid), [0.5], lw=3, colors="k")

def plot_points(x, n_samples):
    c = "bgr"
    m = "xos"
    for i, point in enumerate(x):
        idx = np.random.randint(0, point.shape[0], size = n_samples)
        plt.plot(point[idx, 0], point[idx, 1], c[i] + m[i])

def plot_contours(xx, yy, x_range, y_range, u, sigma):
    nclasses = len(u)
    c = "bgr"
    m = "xos"
    for i in range(nclasses):
        prob = mvn2d(x_range, y_range, u[i], sigma[i])
        cs = plt.contour(xx, yy, prob, colors=c[i])

def make_one_hot(yhat):
    yy = yhat.reshape(-1, 1)  # make 2d
    enc = OneHotEncoder(sparse=False)
    Y = enc.fit_transform(yy)
    return Y

ngrid = 300
n_samples = 30

plt.figure()
plot_points(X, n_samples)
plt.axis("square")
plt.tight_layout()
plt.show()
# %%
