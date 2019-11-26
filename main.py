# import numpy as np
# import pandas as pd
# import cv2
# import os
# import csv
#
# fer_data=pd.read_csv('fer2013.csv',delimiter=',')
#
# def save_fer_img():
#
#     for index,row in fer_data.iterrows():
#         pixels=np.asarray(list(row['pixels'].split(' ')),dtype=np.uint8)
#         img=pixels.reshape((48,48))
#         pathname=os.path.join('fer_images',str(index)+'.jpg')
#         cv2.imwrite(pathname,img)
#         print('image saved ias {}'.format(pathname))
#
#
#
#
# #save_fer_img()
import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

