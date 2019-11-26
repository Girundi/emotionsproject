from __future__ import print_function
import cv2
import numpy as np
from time import sleep
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from keras.preprocessing.image import img_to_array
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dlib
import face_recognition
import pkg_resources

haar_xml = pkg_resources.resource_filename(
    'cv2', 'data/haarcascade_frontalface_default.xml')

face_classifier = cv2.CascadeClassifier(haar_xml)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# classifier = load_model('./model_v6_23.hdf5')
class_labels = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']

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

PATH = "./check_points_4/net_714.pth"
net = Net()
net.load_state_dict(torch.load(PATH))

def face_detector(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), img

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    except:
        return (x, w, y, h), np.zeros((48, 48), np.uint8), img
    return (x, w, y, h), roi_gray, img


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    rect, face, image = face_detector(frame)
    if np.sum([face]) != 0.0:
        roi = 2*(face.astype("float") / 255.0) - 1
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=0)
        roi = torch.from_numpy(roi)
        roi = roi.squeeze(dim=4)
        # make a prediction on the ROI, then lookup the class
        preds = net(roi.to(device))[0]
        label = class_labels[preds.argmax()]
        label_position = (rect[0] + int((rect[1] / 2)), rect[2] + 25)
        cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(image, "No Face Found", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('All', image)
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()