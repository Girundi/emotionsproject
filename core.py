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
        self.sheet_shared = False

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

class Classifier(nn.Module):

    # def __init__(self):
    #     super(Classifier, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 6, 5)  # # # TODO <<"^~^">>
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 24, 5)
    #     self.fc1 = nn.Linear(24 * 9 * 9, 486)
    #     self.fc2 = nn.Linear(486, 120)
    #     self.fc3 = nn.Linear(120, 84)
    #     self.fc4 = nn.Linear(84, 36)
    #     self.fc5 = nn.Linear(36, 7)
    #
    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 24 * 9 * 9)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = F.relu(self.fc3(x))
    #     x = F.relu(self.fc4(x))
    #     x = self.fc5(x)
    #     return x

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

    def __init__(self, input_mode = 0, output_mode = 0, record_video = False,
                 sheet_name = None, email_to_share = None, channel = 0):
        self.save_into_sheet = True
        if sheet_name == None or email_to_share == None:
            self.save_into_sheet = False
        if self.save_into_sheet:
            self.api = API(sheet_name, email_to_share)
        uri = 'rtsp://admin:Supervisor@{}:554/Streaming/Channels/1'
        self.input_mode = input_mode
        self.output_mode = output_mode      # 0 - pretty display, 1 - separate graph
        self.record_video = record_video
        if input_mode == 0:
            self.channel = 0    # webcam
        elif input_mode == 1:         # ip camera
            self.channel = uri.format(channel)
            self.ip = channel
        elif input_mode == 2:         # video
            self.channel = channel




    # from classifier by Sizykh Ivan
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_labels = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']
    # PATH = "./check_points_4/net_714.pth"
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
        # os.chdir(r"frames")
        out_arr = []
        while ret:
            ret, frame = cap.read()
            if time.time() - t >= fps:
                t = time.time()
                out_arr.append(frame)
                # cv2.imwrite("frame " + str(count_frames) + ".jpg", frame)

        return np.asarray(out_arr)

    def make_video(self, frames, num_fps):
        mode = ""
        if self.input_mode == 0:
            mode = "wc_"
        elif self.input_mode == 1:
            mode = self.ip
        elif self.input_mode == 2:
            mode = str(self.channel) # datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        string = "./" + mode + datetime.datetime.now().strftime("%Y-%m-%d") + ".mp4"
        writer = cv2.VideoWriter(
            string,
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

    def run(self, fps_factor):


        last_row = 1
        last_column = 1

        # to load RetinaFace model
        if self.detector is None or self.cfg is None:
            self.prerun()
        table = []
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

        # if x[len(x) - 1] >= img_raw.shape[1] - 50:
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

        # if self.output_mode == 1:
        #     line1, = plt.plot(0, 0, 'ko-')
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
                    img = img.to(self.device)
                    scale = scale.to(self.device)
                    # graph = 0
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
                            ntable = np.asarray(table)
                            x = range(1, len(table) + 1)
                            x = np.asarray(x)
                            shift = 0

                            def softmax(x):
                                return np.exp(x) / sum(np.exp(x))
                            ntable = softmax(ntable)
                            angry_scores = ntable[:, 1]
                            disgust_scores = ntable[:, 2]
                            fear_scores = ntable[:, 3]
                            happy_scores = ntable[:, 4]
                            sad_scores = ntable[:, 5]
                            surprise_scores = ntable[:, 6]
                            neutral_scores = ntable[:, 7]

                            if self.output_mode == 0:
                                ntable = ntable * 50
                                scale = (img_raw.shape[1] - 50) / x[len(x) - 1]
                                x = x * scale

                                shift = img_raw.shape[0] - 40
                                angry_scores = shift - ntable[:, 1]
                                disgust_scores = shift - ntable[:, 2]
                                fear_scores = shift - ntable[:, 3]
                                happy_scores = shift - ntable[:, 4]
                                sad_scores = shift - ntable[:, 5]
                                surprise_scores = shift - ntable[:, 6]
                                neutral_scores = shift - ntable[:, 7]

                                plot = np.vstack((x, angry_scores)).astype(np.int32).T
                                cv2.polylines(img_raw, [plot], isClosed=False, color=(0, 0, 255))
                                cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                                cv2.putText(img_raw, self.class_labels[0], cord, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1 )

                                plot = np.vstack((x, disgust_scores)).astype(np.int32).T
                                cv2.polylines(img_raw, [plot], isClosed=False, color=(0, 255, 0))
                                cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                                cv2.putText(img_raw, self.class_labels[1], cord, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0),1)

                                plot = np.vstack((x, fear_scores)).astype(np.int32).T
                                cv2.polylines(img_raw, [plot], isClosed=False, color=(255, 255, 255))
                                cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                                cv2.putText(img_raw, self.class_labels[2], cord, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),1)

                                plot = np.vstack((x, happy_scores)).astype(np.int32).T
                                cv2.polylines(img_raw, [plot], isClosed=False, color=(0, 255, 255))
                                cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                                cv2.putText(img_raw, self.class_labels[3], cord, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255),1)

                                plot = np.vstack((x, sad_scores)).astype(np.int32).T
                                cv2.polylines(img_raw, [plot], isClosed=False, color=(153, 153, 255))
                                cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                                cv2.putText(img_raw, self.class_labels[4], cord, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (153, 153, 255),1)

                                plot = np.vstack((x, surprise_scores)).astype(np.int32).T
                                cv2.polylines(img_raw, [plot], isClosed=False, color=(153, 0, 76))
                                cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                                cv2.putText(img_raw, self.class_labels[5], cord, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (153, 0, 76),1)

                                plot = np.vstack((x, neutral_scores)).astype(np.int32).T
                                cv2.polylines(img_raw, [plot], isClosed=False, color=(96, 96, 96))
                                cord = (plot[len(plot) - 1][0], plot[len(plot) - 1][1])
                                cv2.putText(img_raw, self.class_labels[6], cord, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (96, 96, 96),1)
                            # elif self.output_mode == 1:
                            #     line1, = plt.plot(x, angry_scores, 'ko-')
                                # figure = plt.figure()
                                # figure.add_subplot(111)
                                # line1.set_ydata(angry_scores)
                                # figure.convas.draw()
                                # graph = np.fromstring(figure.convas.tostring_rgb(), dtype=np.uint8, sep='')
                                # graph = graph.reshape(figure.canvas.get_width_height()[::-1] + (3, ))
                                # graph = cv2.cvtColor(graph, cv2.COLOR_RGB2BGR)




                            # plot = np.vstack((x, table))

                            # dots on facial features

                            # cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                            # cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                            # cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                            # cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                            # cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

                    frames.append(img_raw)
                    if img_raw.shape[1] >= 1000:
                        persent = 50
                        width = int(img_raw.shape[1] * persent / 100)
                        height = int(img_raw.shape[0] * persent / 100)
                        new_shape = (width, height)
                        img_raw = cv2.resize(img_raw, new_shape, interpolation=cv2.INTER_AREA)
                    cv2.imshow('Face Detector', img_raw)

                    if self.save_into_sheet and i % 25 == 0:
                        self.api.write_table(table)

                    if self.output_mode == 1:
                        angry_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        angry_graph.set_ydata(angry_scores[x.shape[0] - 100:x.shape[0] - 1])
                        disgust_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        disgust_graph.set_ydata(disgust_scores[x.shape[0] - 100:x.shape[0] - 1])
                        fear_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        fear_graph.set_ydata(fear_scores[x.shape[0] - 100:x.shape[0] - 1])
                        happy_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        happy_graph.set_ydata(happy_scores[x.shape[0] - 100:x.shape[0] - 1])
                        sad_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        sad_graph.set_ydata(sad_scores[x.shape[0] - 100:x.shape[0] - 1])
                        surprise_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        surprise_graph.set_ydata(surprise_scores[x.shape[0] - 100:x.shape[0] - 1])
                        neutral_graph.set_xdata(x[x.shape[0] - 100:x.shape[0] - 1])
                        neutral_graph.set_ydata(neutral_scores[x.shape[0] - 100:x.shape[0] - 1])


                        figure.canvas.draw()
                        figure.canvas.flush_events()
                        axa = plt.gca()
                        axa.relim()
                        axa.autoscale_view(True, True, True)

                # save image
                #         os.chdir(r"D:/Projects/Vishka/3term/emotionsproject/emotions/detect_frames")
                #         cv2.imwrite("detect_frame " + str(i) + ".jpg", img_raw)

            except:
                if self.channel == 0:
                    cap = cv2.VideoCapture(0)
                else:
                    cap = cv2.VideoCapture(self.channel)
                continue

            if cv2.waitKey(1) == 13:
                break
            i += 1
        cap.release()
        cv2.destroyAllWindows()
        if self.record_video:
            self.make_video(frames, 25/fps_factor)


