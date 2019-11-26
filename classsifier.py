import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import svm
import torch.optim as optim
import time

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

data = np.genfromtxt('fer2013.csv', delimiter=',', dtype=None)

#data = np.delete(data,0)
#data = np.delete(data,0)
#data = np.delete(data,0)

ANGRY    = [1,0,0,0,0,0,0]
HAPPY    = [0,0,0,1,0,0,0]
SAD      = [0,0,0,0,1,0,0]
DISGUST  = [0,1,0,0,0,0,0]
NEUTRAL  = [0,0,0,0,0,0,1]
FEAR     = [0,0,1,0,0,0,0]
SURPRISE = [0,0,0,0,0,1,0]

EMOTIONS = [ANGRY, DISGUST, FEAR, HAPPY, SAD, SURPRISE, NEUTRAL]
classes = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']
train_data = []
test_data = []
usage = []

for i in range(1,35888):

    buf1 = data[i][1].decode("utf-8")
    buf1 = buf1.split(' ')
    buf1 = [int(i) for i in buf1]

    buf0 = int(data[i][0].decode("utf-8"))

    train_data.append(np.asarray(buf1))



    test_data.append(EMOTIONS[buf0])
    usage.append(data[i][2])


train_data = np.asarray(train_data)
test_data = np.asarray(test_data)

print(train_data.shape)
print(test_data.shape)

# #Training
# n_training_samples = 28709
# train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
#
# #Validation
# n_val_samples = 3589
# val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))
#
# #Test
# n_test_samples = 3589
# test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

