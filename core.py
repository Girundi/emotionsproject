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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch
import gspread as gs
from oauth2client.service_account import ServiceAccountCredentials
# import multiprocessing as mp
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def update_vector(row, column, vector, sheet):
#     for em in vector:
#         column += 1
#         sheet.update_cell(row, column, str(em))


class API:

    def __init__(self, name, email_to_share):
        self.scope = ['https://spreadsheets.google.com/feeds',
                      'https://www.googleapis.com/auth/drive']

        self.credentials = ServiceAccountCredentials.from_json_keyfile_name('Emotions Project-481579272f6a.json',
                                                                            self. scope)
        self.client = gs.authorize(self.credentials)
        self.sheet_name = name
        self.email_to_share = email_to_share

    def write_data(self,  vector, timestamp):
        try:
            sheet = self.client.open(self.sheet_name)
        except gs.exceptions.SpreadsheetNotFound:
            sheet = self.client.create(self.sheet_name)
            if isinstance(self.email_to_share, list):
                for email in self.email_to_share:
                    sheet.share(email, perm_type='user', role='writer')
            else:
                sheet.share(self.email_to_share, perm_type='user', role='write')

        row = 1
        column = 1
        sheet = sheet.sheet1
        value = sheet.cell(row, column).value
        if value != "":
            while value != "":
                row += 1
                value = sheet.cell(row, column).value

        if value == "":
            sheet.update_cell(row, column, timestamp)

            # def update_vector(row, column, vector, sheet):
            #     for em in vector:
            #         column += 1
            #         sheet.update_cell(row, column, str(em))

            # pool = mp.Pool(mp.cpu_count())
            for em in vector:
                column += 1
                sheet.update_cell(row, column, str(em))


            # pool.apply(update_vector, args=(row, column, vector, sheet))
            row += 1
            column = 1

        return row, column

    # def write_data(self, sheetname, vector, timestamp, last_row, last_column):
    #     try:
    #         sheet = self.client.open(sheetname)
    #     except gs.exceptions.SpreadsheetNotFound:
    #         sheet = self.client.create(sheetname)
    #         sheet.share('iasizykh@miem.hse.ru', perm_type='user', role='writer')
    #     sheet = sheet.sheet1
    #     sheet.update_cell(last_row, last_column, timestamp)
    #
    #     # pool = mp.Pool(mp.cpu_count())
    #     # pool.apply(update_vector, args=(last_row, last_column, vector, sheet))
    #     # pool.close()
    #     for em in vector:
    #         last_column += 1
    #         sheet.update_cell(last_row, last_column, str(em))
    #     # for em in vector:
    #     #     last_column += 1
    #     #     p = mp.Process(target=sheet.update_cell, args=(last_row, last_column, em))
    #     #     p.start()
    #
    #     last_row += 1
    #     last_column = 1
    #     return last_row, last_column

    def write_table(self, table):
        try:
            sheet = self.client.open(self.sheet_name)
        except gs.exceptions.SpreadsheetNotFound:
            sheet = self.client.create(self.sheet_name)
            if isinstance(self.email_to_share, list):
                for email in self.email_to_share:
                    sheet.share(email, perm_type='user', role='writer')
            else:
                sheet.share(self.email_to_share, perm_type='user', role='write')
        sheet = sheet.sheet1
        cell_list = sheet.range(1, 1, len(table), 8)

        for i in range(len(cell_list) // 8):
            for j in range(8):
                cell_list[i*8 + j].value = table[i][j]
        sheet.update_cells(cell_list)

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(1, 6, 5)  # # # TODO<<"^~^">>TODO
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 24, 5)
        self.fc1 = nn.Linear(24 * 9 * 9, 486)
        self.fc2 = nn.Linear(486, 84)
        # self.fc3 = nn.Linear(120, 84).to(device)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 24 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x


class Emanalisis():

    def __init__(self, sheet_name, email_to_share, cam_ip):
        self.api = API(sheet_name, email_to_share)
        self.ip = cam_ip    # TODO add jay check

    # from classifier by Sizykh Ivan
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_labels = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']
    PATH = "./check_points_4/net_714.pth"
    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load(PATH))

    # from detector by Belyakova Katerina
    parser = argparse.ArgumentParser(description='Retinaface')

    parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

    parser.add_argument('-v', '--video', default='vid.mp4', type=str)

    args = parser.parse_args()

    # let those be, might be used for further improvements

    def load_model(self, model, pretrained_path, load_to_cpu):

        def remove_prefix(state_dict, prefix):
            ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
            print('remove prefix \'{}\''.format(prefix))
            f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
            return {f(key): value for key, value in state_dict.items()}

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

    def load_video(self, video, num_fps):
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

    def make_video(self, count_frames, num_fps):
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

    # to run
    resize = 1
    # ip = '172.18.191.137'
    uri = 'rtsp://admin:Supervisor@{}:554/Streaming/Channels/1'

    def prerun(self):
        torch.set_grad_enabled(False)
        cfg = None
        if self.args.network == "mobile0.25":
            cfg = cfg_mnet
        elif self.args.network == "resnet50":
            cfg = cfg_re50
        # net and model
        detector = RetinaFace(cfg=cfg, phase='test')
        detector = self.load_model(model=detector, pretrained_path=self.args.trained_model, load_to_cpu=self.args.cpu)
        detector.eval()
        print('Finished loading model!')
        print(detector)
        cudnn.benchmark = True
        self.detector = detector.to(self.device)
        self.cfg = cfg

    detector = None
    cfg = None

    def run(self, fps):

        last_row = 1
        last_column = 1

        # to load RetinaFace model
        if self.detector is None or self.cfg is None:
            self.prerun()
        table = []
        if self.ip == 0:
            cap = cv2.VideoCapture(0)#self.uri.format(ip))
        else:
            cap = cv2.VideoCapture(self.uri.format(self.ip))
        i = 0
        while True:

            ret, img_raw = cap.read()
            try:
                if i % fps == 0:
                    img = np.float32(img_raw)

                    im_height, im_width, _ = img.shape
                    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                    img -= (104, 117, 123)
                    img = img.transpose(2, 0, 1)
                    img = torch.from_numpy(img).unsqueeze(0)
                    img = img.to(self.device)
                    scale = scale.to(self.device)

                    tic = time.time()
                    loc, conf, landms = self.detector(img)  # forward pass
                    print('net forward time: {:.4f}'.format(time.time() - tic))

                    priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
                    priors = priorbox.forward()
                    priors = priors.to(self.device)
                    prior_data = priors.data
                    boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
                    boxes = boxes * scale / self.resize
                    boxes = boxes.cpu().numpy()
                    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                    landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
                    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                           img.shape[3], img.shape[2]])
                    scale1 = scale1.to(self.device)
                    landms = landms * scale1 / self.resize
                    landms = landms.cpu().numpy()

                    # ignore low scores
                    inds = np.where(scores > self.args.confidence_threshold)[0]
                    boxes = boxes[inds]
                    landms = landms[inds]
                    scores = scores[inds]

                    # keep top-K before NMS
                    order = scores.argsort()[::-1][:self.args.top_k]
                    boxes = boxes[order]
                    landms = landms[order]
                    scores = scores[order]

                    # do NMS
                    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep = py_cpu_nms(dets, self.args.nms_threshold)
                    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
                    dets = dets[keep, :]
                    landms = landms[keep]

                    # keep top-K faster NMS
                    dets = dets[:self.args.keep_top_k, :]
                    landms = landms[:self.args.keep_top_k, :]

                    dets = np.concatenate((dets, landms), axis=1)

                    # show image
                    for b in dets:
                        if b[4] < self.args.vis_thres:
                            continue
                        text = "{:.4f}".format(b[4])
                        b = list(map(int, b))
                        crop_img = img_raw[b[1]:b[3], b[0]:b[2]]
                        # count_crop = count_crop + 1
                        dim = (48, 48)
                        if crop_img.sum() != 0:
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
                            preds = self.classifier(roi.to(self.device))[0]
                            label = self.class_labels[preds.argmax()]
                            preds = preds.tolist()
                            # i is timestamp for temporal
                            table.append([i, preds[0], preds[1],preds[2], preds[3], preds[4], preds[5], preds[6]])

                            # ###########
                            #
                            # # reference to API
                            # api = API()
                            # last_row, last_column = api.write_data('test1', preds.tolist(), 0, last_row, last_column)
                            #
                            # ######

                            # os.chdir(r"D:/Projects/Vishka/3term/emotionsproject/emotions/crop")
                            # cv2.imwrite("crop_face " + str(count_crop) + ".jpg", gray_res)
                            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                            cx = b[0]
                            cy = b[1] + 12
                            # cv2.putText(img_raw, text, (cx, cy),
                            #             cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                            # label_position = (b[0] + int((b[1] / 2)), b[2] + 25)
                            cv2.putText(img_raw, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                            # dots on facial features

                            # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                            # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                            # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                            # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                            # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)


                    cv2.imshow('Face Detector', img_raw)

                # save image
                #         os.chdir(r"D:/Projects/Vishka/3term/emotionsproject/emotions/detect_frames")
                #         cv2.imwrite("detect_frame " + str(i) + ".jpg", img_raw)

            except:
                if self.ip == 0:
                    cap = cv2.VideoCapture(0)
                else:
                    cap = cv2.VideoCapture(self.uri.format(self.ip))
                continue

            if cv2.waitKey(1) == 13:
                break
            i += 1
        cap.release()
        cv2.destroyAllWindows()
        self.api.write_table(table)

