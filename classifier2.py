from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode
#
# frame = pd.read_csv('fer2013.csv')
# n = 65
# cl = frame.iloc[:, 0]
# data = frame.iloc[:, 1].as_matrix()
# realdata = np.ndarray((35888,2304))

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data = np.genfromtxt('fer2013.csv', delimiter=',', dtype=None)

# ANGRY    = [1,0,0,0,0,0,0]
# HAPPY    = [0,0,0,1,0,0,0]
# SAD      = [0,0,0,0,1,0,0]
# DISGUST  = [0,1,0,0,0,0,0]
# NEUTRAL  = [0,0,0,0,0,0,1]
# FEAR     = [0,0,1,0,0,0,0]
# SURPRISE = [0,0,0,0,0,1,0]
#
# EMOTIONS = [ANGRY, DISGUST, FEAR, HAPPY, SAD, SURPRISE, NEUTRAL]
classes = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']
train_data = []
test_data = []
usage = []


for i in range(1,35888):

    buf1 = data[i][1].decode("utf-8")
    buf1 = buf1.split(' ')
    buf1 = [int(i) for i in buf1]


    buf0 = int(data[i][0].decode("utf-8"))

    train_data.append(np.reshape(buf1,(48,48) ) )

    test_data.append(buf0)#EMOTIONS[buf0])
    usage.append(data[i][2])


train_data = np.asarray(train_data)
test_data = np.asarray(test_data)

[train_data, test1, test2] = np.split(train_data, [28708, 28708+3589])  ## ## ##TODO retrain with bigger set for whole day
[test_data, check1, check2] = np.split(test_data, [28708, 28708+3589])  ## ## ##maybe try some
train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
train_data = train_data.unsqueeze(1)
train_data = train_data.reshape((7177, 4, 1 , 48, 48))
test_data = test_data.reshape((7177, 4))    # , 1))
train_data = train_data.float()
test_data = test_data.long()
train_data = 2*(train_data / 255) - 1   # Normalize
# F.normalize(train_data,out=train_data)
print(type(train_data[0]))
print(train_data[0].shape)
imshow(utils.make_grid(train_data[1]))
train_data = train_data.to(device)
test_data = test_data.to(device)
# F.normalize(test_data,out=test_data)
# norm = transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
# train_data = norm(train_data)
# train_data = transforms.ToPILImage(train_data)
# test_data = transforms.ToPILImage(test_data)
print(train_data.shape)
print(test_data.shape)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5).to(device) # # # TODO <<"^~^">>
        self.pool = nn.MaxPool2d(2, 2).to(device)
        self.conv2 = nn.Conv2d(6, 24, 5).to(device)
        self.fc1 = nn.Linear(24 * 9 * 9, 486).to(device)
        self.fc2 = nn.Linear(486, 84).to(device)
        # self.fc3 = nn.Linear(120, 84).to(device)
        self.fc3 = nn.Linear(84, 7).to(device)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))).to(device)
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 24 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x

net = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print(len(data))

for epoch in range(1000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(7177):
        # get the inputs; data is a list of [inputs, labels]
        inputs = train_data[i].to(device)
        labels = test_data[i].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    PATH = './check_points_4/net_' + str(epoch) + '.pth'
    torch.save(net.state_dict(), PATH)

print('Finished Training')

# PATH = './net.pth'
# torch.save(net.state_dict(), PATH)