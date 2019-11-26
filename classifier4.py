from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import regularizers
from keras.regularizers import l1
from keras.models import load_model
import cv2
from keras.preprocessing.image import img_to_array
import numpy as np

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
train_data = np.expand_dims(train_data, axis=1)
train_data = train_data.reshape((7177, 4, 1, 48, 48))
test_data = test_data.reshape((7177, 4))    # , 1))

test1 = np.delete(test1, 0, 0)
check1 = np.delete(check1, 0, 0)
test1 = np.expand_dims(test1, axis=1)
test1 = np.expand_dims(test1, axis=4)
test1 = test1.astype("float")
check1 = check1.astype("float")
test1 = (test1 / 255.)

test2 = np.delete(test2, 0, 0)
check2 = np.delete(check2, 0, 0)
test2 = np.delete(test2, 0, 0)
check2 = np.delete(check2, 0, 0)
test2 = np.expand_dims(test2, axis=1)
test2 = np.expand_dims(test2, axis=4)
test2 = test2.astype("float")
check2 = check2.astype("float")
test2 = (test2 / 255)


classifier = load_model('./model_v6_23.hdf5')

# img = cv2.imread("fer_images/3.jpg")
#
# img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
#
# roi = img.astype("float")
#
# roi = roi / 255.0
#
# roi = img_to_array(roi)
#
# roi = np.expand_dims(roi, axis=0)
#
# preds = classifier.predict(roi)[0]
#
# print(preds)
total = 0
correct = 0
class_correct = list(0. for i in range(7))
class_total = list(0. for i in range(7))
for i in range(3588):
    outputs = classifier.predict(test1[i])[0]
    outputs = np.argmax(outputs)
    if outputs == check1[i]:
        correct += 1
        class_correct[int(check1[i])] += 1
    class_total[int(check1[i])] += 1
print('Accuracy of the network on the 3588 test images: %d %%' % (
    100 * correct / 3588))
for i in range(7):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
print("\n")

total = 0
correct = 0
class_correct = list(0. for i in range(7))
class_total = list(0. for i in range(7))
for i in range(897):
    outputs = classifier.predict(test2[i])[0]
    outputs = np.argmax(outputs)
    if outputs == check1[i]:
        correct += 1
        class_correct[int(check2[i])] += 1
    class_total[int(check2[i])] += 1
print('Accuracy of the network on the 897 test images: %d %%' % (
    100 * correct / 897))
for i in range(7):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


