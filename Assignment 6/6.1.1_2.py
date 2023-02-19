import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 7)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from statistics import median

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307,0.3081)
])

train_set = datasets.MNIST(root = './data', download = True, train = True, transform = transform)
test_set = datasets.MNIST(root = './data', download = True, train = False, transform = transform)

train_indices, val_indices, _, _ = train_test_split(
    range(len(train_set)),
    train_set.targets,
    stratify=train_set.targets,
    test_size=0.2,
)

train_split = Subset(train_set, train_indices)
val_split = Subset(train_set, val_indices)

train_loader = DataLoader(train_split, batch_size = 1, shuffle = True)
val_loader = DataLoader(val_split, batch_size = 1, shuffle = True)

x_train = []
y_train = []
x_val = []
y_val = []
for img, y in train_loader:
    img = img.cpu().detach().numpy()
    x_train.append(img.ravel())
    y_train.append(y.cpu().detach().numpy())

for img, y in val_loader:
    img = img.cpu().detach().numpy()
    x_val.append(img.ravel())
    y_val.append(y.cpu().detach().numpy())

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

# Median finding with 1000 datapoints
pts = x_train[np.random.randint(0, 48000, 1000), :]
mat = euclidean_distances(pts, pts)
mdian = median(mat.ravel())

print(mdian)

# gamma is 1/var
sigma = [mdian]
C = [0.01, 0.1, 1, 10, 100]

print("Training Phase ---->")
val_acc = {}
for c in C:
    for s in sigma:
        print("C: {} and Sigma: {}".format(c, s))
        model = SVC(C=c, kernel='rbf', gamma=1/s**2)
        model.fit(x_train, y_train.ravel())
        acc = accuracy_score(y_val.ravel(), model.predict(x_val))
        print("Validation Accuracy: {}".format(acc))
        val_acc[(c, s)] = acc

b_acc = max(val_acc.values())
for key in val_acc.keys():
    if val_acc[key] == b_acc:
        best_param = key

print("\nBest Parameter: C = {} , Sigma = {}\n".format(best_param[0], best_param[1]))

test_loader = DataLoader(test_set, batch_size = 1, shuffle = True)

x_test = []
y_test = []

for img, y in test_loader:
    img = img.cpu().detach().numpy()
    x_test.append(img.ravel())
    y_test.append(y.cpu().detach().numpy())

x_train = np.concatenate((x_train, x_val))
y_train = np.concatenate((y_train, y_val))
x_test = np.array(x_test)
y_test = np.array(y_test)

model = SVC(C=best_param[0], kernel='rbf', gamma=1/best_param[1]**2)
model.fit(x_train, y_train.ravel())
acc = accuracy_score(y_test.ravel(), model.predict(x_test))
print("Test Accuracy = {}".format(acc))

