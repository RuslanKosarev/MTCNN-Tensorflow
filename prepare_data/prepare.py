# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import cv2
import numpy as np
from prepare_data.utils import IoU, readlines
from prepare_data.BBox_utils import read_bbox_data, BBox
from prepare_data.Landmark_utils import rotate, flip
from train_models import MTCNN_config
from prepare_data import h5utils


def widerdbase(dbase, seed=None):
    np.random.seed(seed=seed)

    positive = []
    negative = []
    part = []

    with dbase.annotation.open() as f:
        annotations = [a.strip() for a in f]
    num = len(annotations)

    print('number of images {}'.format(num))
    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # don't care
    idx = 0

    for annotation in annotations:
        annotation = annotation.split(' ')

        # image path
        im_path = annotation[0]

        # boxed change to float type
        bbox = list(map(float, annotation[1:]))

        # gt
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

        # load image
        img = cv2.imread(dbase.images.joinpath(im_path + '.jpg').as_posix())
        idx += 1

        height, width, channel = img.shape

        # keep crop random parts, until have 50 negative examples get 50 negative sample from every image
        for i in range(50):
            # neg_num's size [40,min(width, height) / 2],min_size:40
            # size is a random number between 12 and min(width,height)
            size = np.random.randint(12, min(width, height) / 2)
            # top_left coordinate
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            # random crop
            crop_box = np.array([nx, ny, nx + size, ny + size])
            # calculate iou
            iou_values = IoU(crop_box, boxes)

            # crop a part from initial image
            cropped_im = img[ny: ny + size, nx: nx + size, :]

            if np.max(iou_values) < 0.3:
                # resize the cropped image to size 12*12
                resized = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                # Iou with all gts must below 0.3
                filename = dbase.negative.joinpath('{}.jpg'.format(n_idx))
                if not cv2.imwrite(str(filename), resized):
                    raise IOError('file {} has not been written'.format(filename))

                negative.append((os.path.join(filename.parent.name, filename.name), 0, 0, 0, 0, 0))
                n_idx += 1

        # for every bounding boxes
        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            # gt's width
            w = x2 - x1 + 1
            # gt's height
            h = y2 - y1 + 1

            # ignore small faces and those faces has left-top corner out of the image
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 20 or x1 < 0 or y1 < 0:
                continue

            # crop another 5 images near the bounding box if IoU less than 0.5, save as negative samples
            for i in range(5):
                # size of the image to be cropped
                size = np.random.randint(12, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)
                # max can make sure if the delta is a negative number , x1+delta_x >0
                # parameter high of randint make sure there will be intersection between bbox and cropped_box
                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                # max here not really necessary
                nx1 = int(max(0, x1 + delta_x))
                ny1 = int(max(0, y1 + delta_y))
                # if the right bottom point is out of image then skip
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
                iou_values = IoU(crop_box, boxes)

                cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]

                if np.max(iou_values) < 0.3:
                    # resize cropped image to be 12 * 12
                    resized = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                    # Iou with all gts must below 0.3
                    filename = dbase.negative.joinpath('{}.jpg'.format(n_idx))
                    if not cv2.imwrite(str(filename), resized):
                        raise IOError('file {} has not been written'.format(filename))

                    negative.append((os.path.join(filename.parent.name, filename.name), 0,
                                     np.NaN, np.NaN, np.NaN, np.NaN))
                    n_idx += 1

            # generate positive examples and part faces
            for i in range(20):
                # pos and part face size [minsize*0.8,maxsize*1.25]
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                if w < 5:
                    continue
                # print (box)
                delta_x = np.random.randint(-0.2*w, 0.2*w)
                delta_y = np.random.randint(-0.2*h, 0.2*h)

                # show this way: nx1 = max(x1+w/2-size/2+delta_x)
                # x1+ w/2 is the central point, then add offset , then deduct size/2
                # deduct size/2 to make sure that the right bottom corner will be out of
                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                # show this way: ny1 = max(y1+h/2-size/2+delta_y)
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                # offset
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                # crop
                cropped_im = img[ny1: ny2, nx1: nx2, :]

                # resize
                resized = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                iou = IoU(crop_box, box_)
                if iou >= 0.65:
                    filename = dbase.positive.joinpath('{}.jpg'.format(p_idx))
                    if not cv2.imwrite(str(filename), resized):
                        raise IOError('file {} has not been written'.format(filename))

                    positive.append((os.path.join(filename.parent.name, filename.name), 1,
                                     offset_x1, offset_y1, offset_x2, offset_y2))
                    p_idx += 1

                elif iou >= 0.4:
                    filename = dbase.part.joinpath('{}.jpg'.format(d_idx))
                    if not cv2.imwrite(str(filename), resized):
                        raise IOError('file {} has not been written'.format(filename))

                    part.append((os.path.join(filename.parent.name, filename.name), -1,
                                 offset_x1, offset_y1, offset_x2, offset_y2))
                    d_idx += 1

        if idx % 100 == 0:
            print('\r{} images done, positive: {}, negative: {}, part: {}'.format(idx, p_idx, n_idx, d_idx), end='')

    print('\r{} images done, positive: {}, negative: {}, part: {}'.format(idx, p_idx, n_idx, d_idx))

    h5utils.write(dbase.h5out, 'positive', np.array(positive, dtype=MTCNN_config.wider_dtype))
    h5utils.write(dbase.h5out, 'negative', np.array(negative, dtype=MTCNN_config.wider_dtype))
    h5utils.write(dbase.h5out, 'part', np.array(part, dtype=MTCNN_config.wider_dtype))


def lfwdbase(dbase, argument=True, seed=None):
    np.random.seed(seed=seed)

    size = 12
    image_id = 0

    data = read_bbox_data(dbase.annotations)

    idx = 0
    output = []

    # image_path bbox landmark(5*2)
    for (inpfile, bbox, landmarkGt) in data:
        # print imgPath
        f_imgs = []
        f_landmarks = []
        # print(imgPath)

        inpfile = dbase.inpdir.joinpath(inpfile)
        img = cv2.imread(str(inpfile))
        if img is None:
            raise IOError('error to read image {}.'.format(inpfile))

        img_h, img_w, img_c = img.shape
        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
        # get sub-image from bbox
        f_face = img[bbox.top:bbox.bottom + 1, bbox.left:bbox.right + 1]

        # resize the gt image to specified size
        f_face = cv2.resize(f_face, (size, size))
        # initialize the landmark
        landmark = np.zeros((5, 2))

        # normalize land mark by dividing the width and height of the ground truth bounding box
        # landmakrGt is a list of tuples
        for index, one in enumerate(landmarkGt):
            # (( x - bbox.left)/ width of bounding box, (y - bbox.top)/ height of bounding box
            rv = ((one[0] - gt_box[0]) / (gt_box[2] - gt_box[0]), (one[1] - gt_box[1]) / (gt_box[3] - gt_box[1]))
            # put the normalized value into the new list landmark
            landmark[index] = rv

        f_imgs.append(f_face)
        f_landmarks.append(landmark.reshape(10))
        landmark = np.zeros((5, 2))

        if argument:
            idx = idx + 1
            if idx % 100 == 0:
                print('\r{} images done.'.format(idx), end='')

            x1, y1, x2, y2 = gt_box
            # gt's width
            gt_w = x2 - x1 + 1
            # gt's height
            gt_h = y2 - y1 + 1
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            # random shift
            for i in range(10):
                bbox_size = np.random.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x = np.random.randint(-0.2*gt_w, 0.2*gt_w)
                delta_y = np.random.randint(-0.2*gt_h, 0.2*gt_h)
                nx1 = int(max(x1 + gt_w / 2 - bbox_size / 2 + delta_x, 0))
                ny1 = int(max(y1 + gt_h / 2 - bbox_size / 2 + delta_y, 0))

                nx2 = nx1 + bbox_size
                ny2 = ny1 + bbox_size
                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])

                cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
                resized_im = cv2.resize(cropped_im, (size, size))
                # cal iou
                iou = IoU(crop_box, np.expand_dims(gt_box, 0))
                if iou > 0.65:
                    f_imgs.append(resized_im)
                    # normalize
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0] - nx1) / bbox_size, (one[1] - ny1) / bbox_size)
                        landmark[index] = rv
                    f_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = f_landmarks[-1].reshape(-1, 2)
                    bbox = BBox([nx1, ny1, nx2, ny2])

                    # mirror
                    if np.random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        # c*h*w
                        f_imgs.append(face_flipped)
                        f_landmarks.append(landmark_flipped.reshape(10))
                    # rotate
                    if np.random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, bbox.reprojectLandmark(landmark_), 5)
                        # landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        f_imgs.append(face_rotated_by_alpha)
                        f_landmarks.append(landmark_rotated.reshape(10))

                        # flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        f_imgs.append(face_flipped)
                        f_landmarks.append(landmark_flipped.reshape(10))

                        # anti-clockwise rotation
                    if np.random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, bbox.reprojectLandmark(landmark_), -5)
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        f_imgs.append(face_rotated_by_alpha)
                        f_landmarks.append(landmark_rotated.reshape(10))

                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        f_imgs.append(face_flipped)
                        f_landmarks.append(landmark_flipped.reshape(10))

            f_imgs, f_landmarks = np.asarray(f_imgs), np.asarray(f_landmarks)

            for i in range(len(f_imgs)):
                if np.sum(np.where(f_landmarks[i] <= 0, 1, 0)) > 0:
                    continue

                if np.sum(np.where(f_landmarks[i] >= 1, 1, 0)) > 0:
                    continue

                outfile = dbase.outdir.joinpath(dbase.keys[0], '{}.jpg'.format(image_id))
                cv2.imwrite(str(outfile), f_imgs[i])

                output.append(tuple([os.path.join(outfile.parent.name, outfile.name), -2] + f_landmarks[i].tolist()))

                image_id += 1

    print('\n')
    h5utils.write(dbase.h5file, dbase.keys[0], np.array(output, dtype=MTCNN_config.lfw_dtype))


def merge(h5out, wider, lfw):

    keys = ('positive', 'negative', 'part')
    for key in keys:
        data = h5utils.read(wider.h5out, key)
        h5utils.write(h5out, key, data)
        print(key, len(data))

    keys = ('landmark',)
    for key in keys:
        data = h5utils.read(lfw.h5out, key)
        h5utils.write(h5out, key, data)
        print(key, len(data))
