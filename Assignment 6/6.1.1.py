import torch
from torch import nn
from torch.optim import SGD
from torchvision import datasets
from torchvision import transforms
from torchsummary import summary
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Reproducibility
torch.manual_seed(0)

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 7)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

# print("Hey")
# Transforming the data
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

train_split = torch.utils.data.Subset(train_set, train_indices)
val_split = torch.utils.data.Subset(train_set, val_indices)

input_size = 28 * 28 * 1
out_size = 10

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model1 = nn.Sequential(
    nn.Linear(input_size, 256),
    nn.ReLU(),
    nn.Linear(256, out_size)
)

model2 = nn.Sequential(
    nn.Linear(input_size, 512),
    nn.ReLU(),
    nn.Linear(512, 32),
    nn.ReLU(),
    nn.Linear(32, out_size)
)

criterion = nn.CrossEntropyLoss()
epochs = 20
batch_sizes = [10, 100, 1000, 10000]
model_nos = [1, 2]

file = open("logs/log.txt", "w")
# file.write("Model 1 Summary: \n")
# file.writelines(summary(model1, (1,input_size)))
# file.write("\nModel 2 Summary: \n")
# file.writelines(summary(model2, (1, input_size)))
# file.write("\n")
models_loss = {}
for n in model_nos:
    if n == 1:
        model = model1
        print("For Model 1: --->")
        print()
        file.write("For Model 1: --->\n")

    elif n == 2:
        model = model2
        print("For Model 2: --->")
        print()
        file.write("For Model 2: --->\n")

    for batch_size in batch_sizes:
        print()
        print("Batch Size: ", batch_size)
        file.write("\nBatch Size: {} \n".format(batch_size))

        model.apply(init_weights)
        optimizer = SGD(model.parameters(), lr = 0.001)

        train_loader = torch.utils.data.DataLoader(train_split, batch_size = batch_size, shuffle = True)
        val_loader = torch.utils.data.DataLoader(val_split, batch_size = batch_size, shuffle = True)

        for epoch in range(1, epochs+1):
            e_loss = 0
            for images, labels in train_loader:
                images = images.view(images.shape[0], -1)
                optimizer.zero_grad()
                out = model(images)
                # print("Out:{} label{} : ".format(out.shape, type(labels)))
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                e_loss += loss.item()
            print("Epoch:{}  Loss:{}".format(epoch, e_loss/len(train_loader)))
            file.write("Epoch:{}  Loss:{}\n".format(epoch, e_loss/len(train_loader)))

        val_acc = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.view(images.shape[0], -1)
                out = model(images)
                _, prediction = torch.max(out, 1)
                val_acc += torch.sum(prediction == labels)/float(images.shape[0])

        print("Validation Accuracy:{}".format(val_acc/float(len(val_loader))))
        file.write("Validation Accuracy: {}\n".format(val_acc/float(len(val_loader))))
        models_loss[(n, batch_size)] = val_acc/float(len(val_loader))

min_loss = max(models_loss.values())
for key in models_loss.keys():
    if models_loss[key] == min_loss:
        best_param = key
print()
print("Best Parameter: ", best_param)
print()
file.write("\nBest Parameter: {}\n\n".format(best_param))

if best_param[0] == 1:
    model = model1
elif best_param[0] == 2:
    model = model2

train_loader = torch.utils.data.DataLoader(train_set, batch_size = best_param[1], shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = best_param[1], shuffle = True)

model.apply(init_weights)
optimizer = SGD(model.parameters(), lr = 0.001)

print("Best parameter training --->")
file.write("Best parameter training --->\n")
loss_vs_epoch = []
for epoch in range(1, epochs+1):
    e_loss = 0
    for images, labels in train_loader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        e_loss += loss.item()
    print("Epoch:{}  Loss:{}".format(epoch, e_loss))
    file.write("Epoch:{}  Loss:{}\n".format(epoch, e_loss))
    loss_vs_epoch.append(e_loss)

# Plotting Graph
plt.plot(loss_vs_epoch, 'bo-')
plt.title("Training Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.xticks(range(1, epochs+1))
plt.savefig("logs/train_vs_epoch.jpg")
plt.show()

test_acc = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.shape[0], -1)
        out = model(images)
        _, prediction = torch.max(out, 1)
        test_acc += torch.sum(prediction == labels)/float(images.shape[0])

print("Test Set Accuracy: ", test_acc/float(len(test_loader)))
file.write("Test Set Accuracy: {}".format(test_acc/float(len(test_loader))))
file.close()


