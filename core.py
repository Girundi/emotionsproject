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
import datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch
import gspread as gs
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.pyplot as plt
import multiprocessing as mp
import requests
import platform


class API:

    def __init__(self, email_to_share):
        self.scope = ['https://spreadsheets.google.com/feeds',
                      'https://www.googleapis.com/auth/drive']

        self.credentials = ServiceAccountCredentials.from_json_keyfile_name('Emotions Project-481579272f6a.json',
                                                                            self.scope)
        self.client = gs.authorize(self.credentials)
        # self.sheet_name = name
        self.email_to_share = email_to_share
        self.sheet_shared = False
        self.nvr_key = {"key": "99a1dfb5342546319e3d5f4de7150f05"}
        self.nvr_url = 'https://nvr.miem.hse.ru/api/gdrive-upload/504'

    # def write_data(self,  vector, timestamp):
    #     try:
    #         sheet = self.client.open(self.sheet_name)
    #     except gs.exceptions.SpreadsheetNotFound:
    #         sheet = self.client.create(self.sheet_name)
    #         if isinstance(self.email_to_share, list):
    #             for email in self.email_to_share:
    #                 sheet.share(email, perm_type='user', role='writer')
    #         else:
    #             sheet.share(self.email_to_share, perm_type='user', role='write')
    #
    #     row = 1
    #     column = 1
    #     sheet = sheet.sheet1
    #     value = sheet.cell(row, column).value
    #     if value != "":
    #         while value != "":
    #             row += 1
    #             value = sheet.cell(row, column).value
    #
    #     if value == "":
    #         sheet.update_cell(row, column, timestamp)
    #
    #         # def update_vector(row, column, vector, sheet):
    #         #     for em in vector:
    #         #         column += 1
    #         #         sheet.update_cell(row, column, str(em))
    #
    #         # pool = mp.Pool(mp.cpu_count())
    #         for em in vector:
    #             column += 1
    #             sheet.update_cell(row, column, str(em))
    #
    #         row += 1
    #         column = 1
    #
    #     return row, column

    def write_table(self, filename, table):
        try:
            sheet = self.client.open(filename)
        except gs.exceptions.SpreadsheetNotFound:
            sheet = self.client.create(filename)
            if not self.sheet_shared:
                self.sheet_shared = True
                if isinstance(self.email_to_share, list):
                    for email in self.email_to_share:
                        sheet.share(email, perm_type='user', role='writer')
                else:
                    sheet.share(self.email_to_share, perm_type='user', role='writer')
        sheet = sheet.sheet1

        if len(table) != 0:
            cell_list = sheet.range(1, 1, len(table), 8)
            for i in range(len(cell_list) // 8):
                for j in range(8):
                    cell_list[i*8 + j].value = table[i][j]
            sheet.update_cells(cell_list)
    def send_to_nvr(self, filename):
        file = open(filename, 'rb')
        files = {'file': file}
        res = requests.post(self.nvr_url, files=files, headers=self.nvr_key)
        return res.status_code

class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(1, 6, 5)  # # # TODO<<"^~^">>TODO
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 24, 5)
        self.fc1 = nn.Linear(24 * 9 * 9, 486)
        self.fc2 = nn.Linear(486, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 24 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Emanalisis():

    def __init__(self, input_mode = 0, output_mode = 0, record_video = False,
                 email_to_share = None, channel = 0, on_gpu = False,
                 display = False, only_headcount = False, send_to_nvr = False):
        self.save_into_sheet = True
        self.on_gpu = on_gpu
        self.send_to_nvr = send_to_nvr
        if email_to_share == None:
            self.save_into_sheet = False
        if self.save_into_sheet or self.send_to_nvr:
            self.api = API(email_to_share)
        uri = 'rtsp://admin:Supervisor@{}:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif'
        self.input_mode = input_mode
        self.output_mode = output_mode      # 0 - pretty display, 1 - separate graph, 2 - graph with black background
        self.record_video = record_video
        self.display = display
        self.only_headcount = only_headcount
        if input_mode == 0:
            self.channel = 0    # webcam
        elif input_mode == 1:         # ip camera
            self.channel = uri.format(channel)
            self.ip = channel
        elif input_mode == 2:         # video
            self.channel = channel




        # from classifier by Sizykh Ivan

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.class_labels = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']
        # PATH = "./check_points_4/net_714.pth"
        PATH = "./net_714.pth"
        if self.on_gpu:
            self.classifier = Classifier().to(self.device)
            self.classifier.load_state_dict(torch.load(PATH))
        else:
            self.classifier = Classifier()
            self.classifier.load_state_dict(torch.load(PATH, map_location={'cuda:0': 'cpu'}))

        # from detector by Belyakova Katerina
        self.parser = argparse.ArgumentParser(description='Retinaface')

        self.parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                            type=str, help='Trained state_dict file path to open')
        self.parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
        self.parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
        self.parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
        self.parser.add_argument('--top_k', default=5000, type=int, help='top_k')
        self.parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
        self.parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
        self.parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
        self.parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

        self.parser.add_argument('-v', '--video', default='vid.mp4', type=str)

        self.parser_args = self.parser.parse_args()

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
            if self.on_gpu:
                device = torch.cuda.current_device()
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, location: storage.cuda(device))
            else:
                pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, location: storage)
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
        # os.chdir(r"frames")
        out_arr = []
        while ret:
            ret, frame = cap.read()
            if time.time() - t >= fps:
                t = time.time()
                out_arr.append(frame)
                # cv2.imwrite("frame " + str(count_frames) + ".jpg", frame)

        return np.asarray(out_arr)

    def make_video(self, filename, frames, num_fps):
        mode = ""
        if self.input_mode == 0:
            mode = "wc_"
        elif self.input_mode == 1:
            mode = self.ip
        elif self.input_mode == 2:
            mode = str(self.channel) # datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        string = './' + filename + '.mp4'

        writer = cv2.VideoWriter(
                string,
                cv2.VideoWriter_fourcc(*'MP4V'),  # codec
                num_fps,  # fps
                (frames[0].shape[1], frames[0].shape[0]))  # width, height
        for frame in (frames):
            writer.write(frame)
        writer.release()
        cv2.destroyAllWindows()
        return string

    # to run
    resize = 1

    def prerun(self):
        torch.set_grad_enabled(False)
        cfg = None
        if self.parser_args.network == "mobile0.25":
            cfg = cfg_mnet
        elif self.parser_args.network == "resnet50":
            cfg = cfg_re50
        # net and model
        detector = RetinaFace(cfg=cfg, phase='test')
        detector = self.load_model(model=detector, pretrained_path=self.parser_args.trained_model,
                                   load_to_cpu=self.parser_args.cpu)
        detector.eval()
        print('Finished loading model!')
        print(detector)

        if self.on_gpu:
            cudnn.benchmark = True
            self.detector = detector.to(self.device)
        else:
            self.detector = detector
        self.cfg = cfg

    detector = None
    cfg = None

    def run(self, filename, fps_factor=1, stop_time=-1):

        last_row = 1
        last_column = 1

        # to load RetinaFace model
        if self.detector is None or self.cfg is None:
            self.prerun()
        table = []
        if self.record_video:
            frames = []
        cap = cv2.VideoCapture(self.channel)#self.uri.format(ip))
        i = 0
        x = []
        angry_scores = []
        disgust_scores = []
        fear_scores = []
        happy_scores = []
        sad_scores = []
        surprise_scores = []
        neutral_scores = []
        emotions_lapse = []

        start_time = time.time()

        if self.output_mode == 1:
            plt.ion()
            figure = plt.figure()
            ax = figure.add_subplot(111)
            angry_graph, = ax.plot(0, 0, 'r-', label=self.class_labels[0])
            disgust_graph, = ax.plot(0,0, 'g-', label=self.class_labels[1])
            fear_graph, = ax.plot(0,0, 'k-', label=self.class_labels[2])
            happy_graph, = ax.plot(0,0, 'y-', label=self.class_labels[3])
            sad_graph, = ax.plot(0,0, 'c-', label=self.class_labels[4])
            surprise_graph, = ax.plot(0,0,'m-', label=self.class_labels[5])
            neutral_graph, = ax.plot(0,0,'b-', label=self.class_labels[6])
            ax.legend()

        while True:

            ret, img_raw = cap.read()
            try:
                if i % fps_factor == 0:
                    img = np.float32(img_raw)

                    im_height, im_width, _ = img.shape
                    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                    img -= (104, 117, 123)
                    img = img.transpose(2, 0, 1)
                    img = torch.from_numpy(img).unsqueeze(0)
                    if self.on_gpu:
                        img = img.to(self.device)
                        scale = scale.to(self.device)
                    # graph = 0
                    tic = time.time()
                    loc, conf, landms = self.detector(img)  # forward pass
                    print('net forward time: {:.4f}'.format(time.time() - tic))

                    priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
                    priors = priorbox.forward()
                    if self.on_gpu:
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
                    if self.on_gpu:
                        scale1 = scale1.to(self.device)
                    landms = landms * scale1 / self.resize
                    landms = landms.cpu().numpy()

                    # ignore low scores
                    inds = np.where(scores > self.parser_args.confidence_threshold)[0]
                    boxes = boxes[inds]
                    landms = landms[inds]
                    scores = scores[inds]

                    # keep top-K before NMS
                    order = scores.argsort()[::-1][:self.parser_args.top_k]
                    boxes = boxes[order]
                    landms = landms[order]
                    scores = scores[order]

                    # do NMS
                    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep = py_cpu_nms(dets, self.parser_args.nms_threshold)
                    dets = dets[keep, :]
                    landms = landms[keep]

                    # keep top-K faster NMS
                    dets = dets[:self.parser_args.keep_top_k, :]
                    landms = landms[:self.parser_args.keep_top_k, :]

                    dets = np.concatenate((dets, landms), axis=1)

                    display_img = np.copy(img_raw)
                    head_count = 0
                    emotions_count = np.zeros(7)

                    # show image
                    for b in dets:
                        if b[4] < self.parser_args.vis_thres:
                            continue
                        text = "{:.4f}".format(b[4])
                        b = list(map(int, b))
                        crop_img = img_raw[b[1]:b[3], b[0]:b[2]]
                        dim = (48, 48)
                        if crop_img.sum() != 0:
                            head_count = head_count + 1
                            if not self.only_headcount:

                                resized = cv2.resize(crop_img, dim)
                                gray_res = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                                ### from webcam
                                roi = gray_res
                                roi = np.array(roi)
                                roi = 2 * (roi.astype("float") / 255.0) - 1

                                roi = np.expand_dims(roi, axis=2)
                                roi = np.expand_dims(roi, axis=0)
                                roi = np.expand_dims(roi, axis=0)
                                roi = torch.from_numpy(roi)
                                roi = roi.float()
                                roi = roi.squeeze(dim=4)
                                # make a prediction on the ROI, then lookup the class
                                tic = time.time()
                                if self.on_gpu:
                                    preds = self.classifier(roi.to(self.device))[0]
                                else:
                                    preds = self.classifier(roi)[0]
                                print(str(time.time() - tic) + " to classify")
                                label = self.class_labels[preds.argmax()]

                                emotions_count[preds.argmax()] += 1

                                preds = preds.tolist()
                                # i is timestamp for temporal

                                table.append([i, preds[0], preds[1],preds[2], preds[3], preds[4], preds[5], preds[6]])
                            if self.output_mode != 2:
                                cv2.rectangle(display_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                                cx = b[0]
                                cy = b[1] + 12
                                if not self.only_headcount:
                                    cv2.putText(display_img, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    if self.output_mode == 2:
                        display_img = np.zeros_like(img_raw)
                        display_img = np.resize(display_img, (400, 1920, 3))
                    emotions_lapse.append(emotions_count.tolist())
                    if not self.only_headcount:
                        ntable = np.asarray(table)
                        shift = 0
                        nemotions_lapse = np.asarray(emotions_lapse) * display_img.shape[0] / (2*head_count)
                        x = range(1, len(emotions_lapse) + 1)
                        x = np.asarray(x)

                        def softmax(x):
                            return np.exp(x) / sum(np.exp(x))
                        ntable = softmax(ntable)
                        unchanged_angry_scores = np.asarray(emotions_lapse)[:, 0]
                        unchanged_disgust_scores = np.asarray(emotions_lapse)[:, 1]
                        unchanged_fear_scores = np.asarray(emotions_lapse)[:, 2]
                        unchanged_happy_scores = np.asarray(emotions_lapse)[:, 3]
                        unchanged_sad_scores = np.asarray(emotions_lapse)[:, 4]
                        unchanged_surprise_scores = np.asarray(emotions_lapse)[:, 5]
                        unchanged_neutral_scores = np.asarray(emotions_lapse)[:, 6]

                        # attention_coef = np.max(np.max(np.flip(emotions_lapse)[0,:],axis=0)) / head_count
                        attention_coef = (np.max(emotions_count)) / head_count
                        # attention_coef = np.mean(emotions_count) / np.max(emotions_count)

                    if self.output_mode == 0 or self.output_mode == 2:
                        if not self.only_headcount:
                            if self.output_mode == 0:
                                ntable = ntable * 50
                                # nemotions_lapse = nemotions_lapse * 5
                            else:
                                ntable = ntable * 300
                                # nemotions_lapse = nemotions_lapse * 30
                            scale = (display_img.shape[1] - 30) / (30*60*25/fps_factor)
                            x = x * scale + 15


                            shift = display_img.shape[0] / 2 - 10
                            # angry_scores = shift + nemotions_lapse[:, 0]
                            # disgust_scores = shift + nemotions_lapse[:, 1]
                            # fear_scores = shift + nemotions_lapse[:, 2]
                            # happy_scores = shift - nemotions_lapse[:, 3]
                            # sad_scores = shift + nemotions_lapse[:, 4]
                            # surprise_scores = shift - nemotions_lapse[:, 5]
                            # neutral_scores = shift - nemotions_lapse[:, 6]

                            possitive_scores = shift - nemotions_lapse[:,3] - nemotions_lapse[:,5] - \
                                               nemotions_lapse[:,6]
                            negative_scores = shift + nemotions_lapse[:, 0] + nemotions_lapse[:, 1] + \
                                              nemotions_lapse[:, 2] + nemotions_lapse[:, 4]
                            possitive_sum = unchanged_happy_scores + unchanged_surprise_scores + \
                                            unchanged_neutral_scores
                            negative_sum = unchanged_angry_scores + unchanged_disgust_scores + unchanged_fear_scores \
                                           + unchanged_sad_scores

                            plot = np.vstack((x, possitive_scores)).astype(np.int32).T
                            cv2.polylines(display_img, [plot], isClosed=False,thickness=2, color=(0, 0, 255))
                            cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                            cv2.putText(display_img,
                                        "possitive " + str(int(np.flip(possitive_sum)[0])),
                                        cord, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1 )

                            plot = np.vstack((x, negative_scores)).astype(np.int32).T
                            cv2.polylines(display_img, [plot], isClosed=False,thickness=2, color=(0, 255, 0))
                            cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                            cv2.putText(display_img,
                                        "negative " + str(int(np.flip(negative_sum)[0]))
                                        , cord, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),1)

                            # plot = np.vstack((x, fear_scores)).astype(np.int32).T
                            # cv2.polylines(display_img, [plot], isClosed=False,thickness=2, color=(255, 255, 255))
                            # cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                            # cv2.putText(display_img,
                            #             self.class_labels[2] + " " + str(int(np.flip(unchanged_fear_scores)[0]))
                            #             , cord, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),1)
                            #
                            # plot = np.vstack((x, happy_scores)).astype(np.int32).T
                            # cv2.polylines(display_img, [plot], isClosed=False,thickness=2, color=(0, 255, 255))
                            # cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                            # cv2.putText(display_img,
                            #             self.class_labels[3] + " " + str(int(np.flip(unchanged_happy_scores)[0]))
                            #             , cord, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255),1)
                            #
                            # plot = np.vstack((x, sad_scores)).astype(np.int32).T
                            # cv2.polylines(display_img, [plot], isClosed=False,thickness=2, color=(153, 153, 255))
                            # cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                            # cv2.putText(display_img,
                            #             self.class_labels[4] + " " + str(int(np.flip(unchanged_sad_scores)[0]))
                            #             , cord, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (153, 153, 255),1)
                            #
                            # plot = np.vstack((x, surprise_scores)).astype(np.int32).T
                            # cv2.polylines(display_img, [plot], isClosed=False,thickness=2, color=(153, 0, 76))
                            # cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                            # cv2.putText(display_img,
                            #             self.class_labels[5] + " " + str(int(np.flip(unchanged_surprise_scores)[0]))
                            #             , cord, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (153, 0, 76),1)
                            #
                            # plot = np.vstack((x, neutral_scores)).astype(np.int32).T
                            # cv2.polylines(display_img, [plot], isClosed=False,thickness=2, color=(96, 96, 96))
                            # cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                            # cv2.putText(display_img,
                            #             self.class_labels[6] + " " + str(int(np.flip(unchanged_neutral_scores)[0]))
                            #             , cord, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (96, 96, 96),1)


                    cv2.putText(display_img, "Head count: " + str(head_count),
                                (5,30), cv2.FONT_HERSHEY_TRIPLEX, 1, (60,20,220))
                    cv2.putText(display_img, "Attention coef: " + str(round(attention_coef, 2)),
                                (400, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (60, 20, 220))





                            # plot = np.vstack((x, table))

                            # dots on facial features

                            # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                            # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                            # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                            # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                            # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

                    if self.record_video and self.output_mode != 1:
                        frames.append(display_img)
                    if display_img.shape[1] >= 1000:
                        persent = 50
                        width = int(display_img.shape[1] * persent / 100)
                        height = int(display_img.shape[0] * persent / 100)
                        new_shape = (width, height)
                        display_img = cv2.resize(display_img, new_shape, interpolation=cv2.INTER_AREA)
                    if self.display:
                        cv2.imshow('Face Detector', display_img)

                    if self.save_into_sheet and i % fps_factor == 0:
                        self.api.write_table(filename, table)

                    if self.output_mode == 1:
                        angry_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        angry_graph.set_ydata(unchanged_angry_scores[x.shape[0] - 100:x.shape[0] - 1])
                        disgust_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        disgust_graph.set_ydata(unchanged_disgust_scores[x.shape[0] - 100:x.shape[0] - 1])
                        fear_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        fear_graph.set_ydata(unchanged_fear_scores[x.shape[0] - 100:x.shape[0] - 1])
                        happy_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        happy_graph.set_ydata(unchanged_happy_scores[x.shape[0] - 100:x.shape[0] - 1])
                        sad_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        sad_graph.set_ydata(unchanged_sad_scores[x.shape[0] - 100:x.shape[0] - 1])
                        surprise_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        surprise_graph.set_ydata(unchanged_surprise_scores[x.shape[0] - 100:x.shape[0] - 1])
                        neutral_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        neutral_graph.set_ydata(unchanged_neutral_scores[x.shape[0] - 100:x.shape[0] - 1])


                        figure.canvas.draw()
                        figure.canvas.flush_events()
                        axa = plt.gca()
                        axa.relim()
                        axa.autoscale_view(True, True, True)
                        if self.record_video:
                            figure.tight_layout(pad=0)
                            axa.margins(0)
                            plot = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
                            plot = plot.reshape(figure.canvas.get_width_height()[::-1] + (3, ))
                            frames.append(plot)

            except:
                cap = cv2.VideoCapture(self.channel)
                continue

            if cv2.waitKey(1) == 13 or time.time() - start_time >= stop_time:
                break
            i += 1
        cap.release()
        cv2.destroyAllWindows()
        plt.close()
        if self.record_video:
            video = self.make_video(filename, frames, 25/fps_factor)
            if self.send_to_nvr:
                self.api.send_to_nvr(video)

