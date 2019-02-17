# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np
import cv2
from prepare_data import wider
from train_models import pnet as pnet
from train_models import rnet as rnet

from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
from prepare_data.loader import TestLoader
from prepare_data import ioutils, h5utils
from prepare_data.utils import convert_to_square
from prepare_data.data_utils import IoU


def generate(h5file, dbwider, model, mode='PNet', threshold=(0.6, 0.6, 0.7), min_face_size=25,
             stride=2, slide_window=False):

    detectors = [None, None, None]

    # load P-Net model
    output_image_size = rnet.Config.image_size
    model_path = '{}-{}'.format(model.path[0], model.epochs[0])
    if slide_window:
        detectors[0] = Detector(pnet.Graph, pnet.Config.image_size, model.batch_size[0], model_path)
    else:
        detectors[0] = FcnDetector(pnet.Graph, model_path)

    # load R-Net model
    if mode.lower() == 'rnet':
        output_image_size = pnet.Config.image_size
        model_path = '{}-{}'.format(model.path[1], model.epochs[1])
        detectors[1] = Detector(rnet.Graph, rnet.Config.image_size, model.batch_size[1], model_path)

    # load onet model
    # if mode.lower() is 'onet':
    #     detectors[1] = Detector(RNet, 24, batch_size[1], model_path[1])
    #     detectors[2] = Detector(ONet, 48, batch_size[2], model_path[2])

    # basedir = '../data/'
    # filename = '../data/WIDER_train/wider_face_train_bbx_gt.txt'
    data = ioutils.read_annotation(dbwider.wider_face_train_bbx_gt)

    files = []
    for file in data['images']:
        files.append(str(dbwider.dir.joinpath(file)))
    data['images'] = files

    # data['images'] = data['images'][:30]
    # data['bboxes'] = data['bboxes'][:30]

    detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                             stride=stride, threshold=threshold, slide_window=slide_window)

    # test_data = TestLoader(data['images'])
    loader = ioutils.ImageLoader(data['images'])
    detections, landmarks = detector.detect_face(loader)

    save_examples(h5file, data, detections, output_image_size)


def save_examples(h5file, data, det_boxes, image_size):

    if not h5file.parent.exists():
        h5file.parent.mkdir()

    for key in ('positive', 'negative', 'part'):
        outdir = h5file.parent.joinpath(key)
        if not outdir.exists():
            outdir.mkdir()

    positive = []
    negative = []
    part = []

    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']
    number_of_images = len(im_idx_list)

    if len(det_boxes) is not number_of_images:
        raise ValueError('incorrect input data')

    # index of neg, pos and part face, used as their image names
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0

    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = ioutils.read_image(im_idx)
        # img = cv2.imread(im_idx)

        # change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            # save negative images and write label Iou with all gts must below 0.3
            if np.max(Iou) < 0.3 and neg_num < 60:
                key_name = os.path.join('negative', '{}.jpg'.format(n_idx))
                ioutils.write_image(resized_im, key_name, prefix=h5file.parent)
                negative.append((key_name, 0, 0, 0, 0, 0))
                n_idx += 1
                neg_num += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    key_name = os.path.join('positive', '{}.jpg'.format(p_idx))
                    ioutils.write_image(resized_im, key_name, prefix=h5file.parent)
                    positive.append((key_name, 1, offset_x1, offset_y1, offset_x2, offset_y2))
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    key_name = os.path.join('part', '{}.jpg'.format(d_idx))
                    ioutils.write_image(resized_im, key_name, prefix=h5file.parent)
                    part.append((key_name, -1, offset_x1, offset_y1, offset_x2, offset_y2))
                    d_idx += 1

    h5utils.write(h5file, 'positive', np.array(positive, dtype=wider.dtype))
    h5utils.write(h5file, 'negative', np.array(negative, dtype=wider.dtype))
    h5utils.write(h5file, 'part', np.array(part, dtype=wider.dtype))
