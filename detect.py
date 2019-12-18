from __future__ import print_function
import os
import argparse
import torch.backends.cudnn as cudnn
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
from nms import nms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch
import gspreed as gs


parser = argparse.ArgumentParser(description='Retinaface')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
# parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

parser.add_argument('-v', '--video', default='vid.mp4', type = str)

args = parser.parse_args()

###from webcam

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
classifier = Net()
classifier.load_state_dict(torch.load(PATH))

#############

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def load_video(video, num_fps):
    fps = 1 / num_fps
    cap = cv2.VideoCapture(video)
    # cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    t = time.time()
    ret = True
    count_frames = 0
    os.chdir(r"frames")

    while ret:
        ret, frame = cap.read()
        if time.time() - t >= fps:
            t = time.time()
            cv2.imwrite("frame " + str(count_frames) + ".jpg", frame)
            count_frames += 1
    return count_frames


def make_video(count_frames, num_fps):
    filelist = []
    for i in range(count_frames):
        image_path = ("D:/Projects/Vishka/3term/emotionsproject/emotions/detect_frames/detect_frame %d.jpg" % i)
        filelist.append(image_path)
    frames = [cv2.imread(fname) for fname in filelist]

    writer = cv2.VideoWriter(
        'output.mp4',
        cv2.VideoWriter_fourcc(*'MP4V'),  # codec
        num_fps,  # fps
        (frames[0].shape[1], frames[0].shape[0]))  # width, height
    for frame in (frames):
        writer.write(frame)
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    net = net.to(device)

    resize = 1

    # make frames from video
    # video = 'vid.mp4'
    # num_fps = 25
    # count_frames = load_video(video, num_fps)

    # testing begin
    # for i in range(count_frames):
    # count_crop = 0
    # image_path = ("D:/Projects/Vishka/3term/emotionsproject/emotions/frames/frame %d.jpg" % i)
    ip = '172.18.191.137'
    cap = cv2.VideoCapture('rtsp://admin:Supervisor@{}:554/Streaming/Channels/1'.format(ip))
    i = 0
    while True:

        ret, img_raw = cap.read()
        try:
            if i % 10 == 0:
                img = np.float32(img_raw)

                im_height, im_width, _ = img.shape
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)
                scale = scale.to(device)

                tic = time.time()
                loc, conf, landms = net(img)  # forward pass
                print('net forward time: {:.4f}'.format(time.time() - tic))

                priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                priors = priorbox.forward()
                priors = priors.to(device)
                prior_data = priors.data
                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                boxes = boxes * scale / resize
                boxes = boxes.cpu().numpy()
                scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
                scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                       img.shape[3], img.shape[2]])
                scale1 = scale1.to(device)
                landms = landms * scale1 / resize
                landms = landms.cpu().numpy()

                # ignore low scores
                inds = np.where(scores > args.confidence_threshold)[0]
                boxes = boxes[inds]
                landms = landms[inds]
                scores = scores[inds]

                # keep top-K before NMS
                order = scores.argsort()[::-1][:args.top_k]
                boxes = boxes[order]
                landms = landms[order]
                scores = scores[order]

                # do NMS
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                keep = py_cpu_nms(dets, args.nms_threshold)
                #keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
                dets = dets[keep, :]
                landms = landms[keep]

                # keep top-K faster NMS
                dets = dets[:args.keep_top_k, :]
                landms = landms[:args.keep_top_k, :]

                dets = np.concatenate((dets, landms), axis=1)

                # show image
                for b in dets:
                    if b[4] < args.vis_thres:
                        continue
                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))
                    crop_img = img_raw[b[1]:b[3], b[0]:b[2]]
                    # count_crop = count_crop + 1
                    dim = (48, 48)
                    if not crop_img.all():
                        resized = cv2.resize(crop_img, dim)
                        gray_res = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                        ### from webcam
                        roi = gray_res
                        roi = np.array(roi)
                        roi = 2 * (roi.astype("float") / 255.0) - 1

                        roi = np.expand_dims(roi, axis=2)
                        # roi = np.transpose(roi, (2, 0, 1))
                        # roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)
                        roi = np.expand_dims(roi, axis=0)
                        roi = torch.from_numpy(roi)
                        roi = roi.float()
                        roi = roi.squeeze(dim=4)
                        # make a prediction on the ROI, then lookup the class
                        preds = classifier(roi.to(device))[0]
                        label = class_labels[preds.argmax()]
                        ###########

                        # os.chdir(r"D:/Projects/Vishka/3term/emotionsproject/emotions/crop")
                        # cv2.imwrite("crop_face " + str(count_crop) + ".jpg", gray_res)
                        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                        cx = b[0]
                        cy = b[1] + 12
                        # cv2.putText(img_raw, text, (cx, cy),
                        #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                        # label_position = (b[0] + int((b[1] / 2)), b[2] + 25)
                        cv2.putText(img_raw, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                cv2.imshow('Fece Detector', img_raw)

                        # dots on faces
            #                 cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            #                 cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            #                 cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            #                 cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            #                 cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            #         os.chdir(r"D:/Projects/Vishka/3term/emotionsproject/emotions/detect_frames")
            #         cv2.imwrite("detect_frame " + str(i) + ".jpg", img_raw)

        except:
            cap = cv2.VideoCapture('rtsp://admin:Supervisor@{}:554/Streaming/Channels/1'.format(ip))
            continue

        if cv2.waitKey(1) == 13:
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()
# make_video(count_frames, num_fps)