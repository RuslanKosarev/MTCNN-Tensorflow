# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import sys
import cv2
import numpy as np
import random
import pathlib as lib
from prepare_data.utils import IoU, readlines
from prepare_data.BBox_utils import read_bbox_data, BBox
from prepare_data.Landmark_utils import rotate, flip


def generate12(input, output):

    fpos = open(output.postxt.as_posix(), 'w')
    fneg = open(output.negtxt.as_posix(), 'w')
    fpart = open(output.parttxt.as_posix(), 'w')

    with input.annotation.open() as f:
        annotations = [a.strip() for a in f]
    num = len(annotations)

    print('number of pictures {}'.format(num))
    p_idx = 0  # positive
    n_idx = 0  # negative
    d_idx = 0  # don't care
    idx = 0
    box_idx = 0

    for annotation in annotations:
        annotation = annotation.split(' ')

        # image path
        im_path = annotation[0]

        # boxed change to float type
        bbox = list(map(float, annotation[1:]))

        # gt
        boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

        # load image
        img = cv2.imread(input.images.joinpath(im_path + '.jpg').as_posix())
        idx += 1

        height, width, channel = img.shape

        neg_num = 0

        # keep crop random parts, until have 50 negative examples get 50 negative sample from every image
        while neg_num < 50:
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
                save_file = output.negdir.joinpath('{}.jpg'.format(n_idx))
                if not cv2.imwrite(save_file.as_posix(), resized):
                    raise IOError('file {} has not been written'.format(save_file))

                text = os.path.join(output.negdir.name, save_file.name) + ' 0\n'
                fneg.write(text)
                n_idx += 1
                neg_num += 1

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
                    save_file = output.negdir.joinpath('{}.jpg'.format(n_idx))
                    if not cv2.imwrite(save_file.as_posix(), resized):
                        raise IOError('file {} has not been written'.format(save_file))

                    text = os.path.join(output.negdir.name, save_file.name) + ' 0\n'
                    fneg.write(text)

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
                    save_file = output.posdir.joinpath('{}.jpg'.format(p_idx))
                    if not cv2.imwrite(save_file.as_posix(), resized):
                        raise IOError('file {} has not been written'.format(save_file))

                    text = os.path.join(output.posdir.name, save_file.name) + \
                        ' 1 {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(p_idx, offset_x1, offset_y1, offset_x2, offset_y2)
                    fpos.write(text)
                    p_idx += 1
                elif iou >= 0.4:
                    save_file = output.partdir.joinpath('{}.jpg'.format(d_idx))
                    if not cv2.imwrite(save_file.as_posix(), resized):
                        raise IOError('file {} has not been written'.format(save_file))

                    text = os.path.join(output.partdir.name, save_file.name) + \
                        ' -1 {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(d_idx, offset_x1, offset_y1, offset_x2, offset_y2)
                    fpart.write(text)
                    d_idx += 1

            box_idx += 1

        if idx % 100 == 0:
            sys.stdout.write('\r{} images done, pos: {} part: {} neg: {}'.format(idx, p_idx, d_idx, n_idx))
        sys.stdout.flush()

    print('{} images done, pos: {} part: {} neg: {}'.format(idx, p_idx, d_idx, n_idx))

    fpos.close()
    fneg.close()
    fpart.close()


def GenerateData(input, output, net, argument=False):
    '''

    :param input: name/path of the text file that contains image path,
                bounding box, and landmarks

    :param output: path of the output dir
    :param net: one of the net in the cascaded networks
    :param argument: apply augmentation or not
    :return:  images and related landmarks
    '''

    size = 12
    image_id = 0

    data = read_bbox_data(input.annotations)

    idx = 0
    f = open(output.annotations.as_posix(), 'w')

    # image_path bbox landmark(5*2)
    for (imgPath, bbox, landmarkGt) in data:
        # print imgPath
        F_imgs = []
        F_landmarks = []
        # print(imgPath)
        img = cv2.imread(imgPath.as_posix())

        if img is None:
            raise IOError('image {} is None'.format(imgPath))

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

        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(10))
        landmark = np.zeros((5, 2))
        if argument:
            idx = idx + 1
            if idx % 100 == 0:
                print(idx, "images done")
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
                    F_imgs.append(resized_im)
                    # normalize
                    for index, one in enumerate(landmarkGt):
                        rv = ((one[0] - nx1) / bbox_size, (one[1] - ny1) / bbox_size)
                        landmark[index] = rv
                    F_landmarks.append(landmark.reshape(10))
                    landmark = np.zeros((5, 2))
                    landmark_ = F_landmarks[-1].reshape(-1, 2)
                    bbox = BBox([nx1, ny1, nx2, ny2])

                    # mirror
                    if random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        # c*h*w
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))
                    # rotate
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, bbox.reprojectLandmark(landmark_), 5)
                        # landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))

                        # flip
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))

                        # anti-clockwise rotation
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                         bbox.reprojectLandmark(landmark_), -5)  # 顺时针旋转
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        F_imgs.append(face_rotated_by_alpha)
                        F_landmarks.append(landmark_rotated.reshape(10))

                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        F_imgs.append(face_flipped)
                        F_landmarks.append(landmark_flipped.reshape(10))

            F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)

            for i in range(len(F_imgs)):
                if np.sum(np.where(F_landmarks[i] <= 0, 1, 0)) > 0:
                    continue

                if np.sum(np.where(F_landmarks[i] >= 1, 1, 0)) > 0:
                    continue

                outfile = output.files.joinpath('{}.jpg'.format(image_id))
                cv2.imwrite(outfile.as_posix(), F_imgs[i])

                text = os.path.join(output.files.name, '{}.jpg'.format(image_id)) + ' -2'
                for l in F_landmarks[i]:
                    text += ' ' + str(l)
                text += '\n'

                f.write(text)

                image_id += 1

    f.close()


def gen_imglist(data_dir):

    size = 12
    net = 'PNet'

    # with open(data_dir.joinpath('pos_12.txt').as_posix(), 'r') as f:
    #     pos = f.readlines()

    pos = readlines(data_dir.joinpath('pos_12.txt'), strip=False)
    neg = readlines(data_dir.joinpath('neg_12.txt'), strip=False)
    part = readlines(data_dir.joinpath('part_12.txt'), strip=False)
    landmark = readlines(data_dir.joinpath('landmark_12_aug.txt'), strip=False)

    filename = data_dir.joinpath('train_landmark.txt')
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True)

    with filename.open('w') as f:
        nums = [len(neg), len(pos), len(part)]
        ratio = [3, 1, 1]

        # base_num = min(nums)
        base_num = 250000
        print(len(neg), len(pos), len(part), base_num)

        def write_output(fout, samples, size):
            for sample in np.random.choice(samples, size=size, replace=True):
                fout.write(sample)

        write_output(f, neg, min([3*base_num, len(neg)]))
        write_output(f, pos, base_num)
        write_output(f, part, base_num)

        for item in landmark:
            f.write(item)
