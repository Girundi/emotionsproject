import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5).to(device) # # # TODO<<"^~^">>TODO
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

classes = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']

data = np.genfromtxt('fer2013.csv', delimiter=',', dtype=None)
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

[train_data, test1, test2] = np.split(train_data, [28708, 28708+3589])
[test_data, check1, check2] = np.split(test_data, [28708, 28708+3589])
train_data = torch.from_numpy(train_data)
test_data = torch.from_numpy(test_data)
train_data = train_data.unsqueeze(1)
train_data = train_data.reshape((7177, 4, 1, 48, 48))
test_data = test_data.reshape((7177, 4))    # , 1))
train_data = train_data.float()
test_data = test_data.long()
F.normalize(train_data,out=train_data)
train_data = train_data.to(device)
test_data = test_data.to(device)

test1 = np.delete(test1, 0, 0)
check1 = np.delete(check1, 0, 0)
test1 = torch.from_numpy(test1)
check1 = torch.from_numpy(check1)
test1 = test1.unsqueeze(1)
test1 = test1.reshape((897, 4, 1, 48, 48))
check1 = check1.reshape((897, 4))
test1 = test1.float()
check1 = check1.long()
test1 = 2*(test1 / 255) - 1

test2 = np.delete(test2, 0, 0)
check2 = np.delete(check2, 0, 0)
test2 = np.delete(test2, 0, 0)
check2 = np.delete(check2, 0, 0)
test2 = torch.from_numpy(test2)
check2 = torch.from_numpy(check2)
test2 = test2.unsqueeze(1)
test2 = test2.reshape((897, 4, 1, 48, 48))
check2 = check2.reshape((897, 4))
test2 = test2.float()
check2 = check2.long()
test2 = 2*(test2 / 255) - 1

print(test2.shape)
imshow(torchvision.utils.make_grid(test1[0]))

test1 = test1.to(device)
check1 = check1.to(device)
test2 = test2.to(device)
check2 = check2.to(device)

####

PATH = "./check_points_4/net_714.pth"
net = Net()
net.load_state_dict(torch.load(PATH))





correct = 0
total = 0
with torch.no_grad():
    for i in range(897):
        images = test1[i].to(device)
        labels = check1[i].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 3588 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(7))
class_total = list(0. for i in range(7))
with torch.no_grad():
    for i in range(897):
        images = test1[i].to(device)
        labels = check1[i].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for j in range(4):
            label = labels[j]
            class_correct[label] += c[j].item()
            class_total[label] += 1


for i in range(7):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

