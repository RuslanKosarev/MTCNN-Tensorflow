# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import cv2
import numpy as np
from prepare_data.utils import IoU


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

                text = os.path.join(output.negdir.name, '{}.jpg 0\n'.format(n_idx))
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

                    text = os.path.join(output.negdir.name, '{}.jpg 0\n'.format(n_idx))
                    fneg.write(text)

                    n_idx += 1

            # generate positive examples and part faces

            for i in range(20):
                # pos and part face size [minsize*0.8,maxsize*1.25]
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                if w < 5:
                    print(w)
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
                    save_file = output.posdir.joinpath('{}.jpg'.format(idx))
                    if not cv2.imwrite(save_file.as_posix(), resized):
                        raise IOError('file {} has not been written'.format(save_file))

                    text = os.path.join(output.posdir.name, '{}.jpg'.format(p_idx)) + \
                        ' 1 {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(p_idx, offset_x1, offset_y1, offset_x2, offset_y2)
                    fpos.write(text)
                    p_idx += 1
                elif iou >= 0.4:
                    save_file = output.partdir.joinpath('{}.jpg'.format(d_idx))
                    if not cv2.imwrite(save_file.as_posix(), resized):
                        raise IOError('file {} has not been written'.format(save_file))

                    text = os.path.join(output.partdir.name, '{}.jpg'.format(d_idx)) + \
                        ' -1 {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(d_idx, offset_x1, offset_y1, offset_x2, offset_y2)
                    fpart.write(text)
                    d_idx += 1

            box_idx += 1
            if idx % 100 == 0:
                print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

    print('{} images done, pos: {} part: {} neg: {}'.format(idx, p_idx, d_idx, n_idx))

    fpos.close()
    fneg.close()
    fpart.close()
