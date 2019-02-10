# coding:utf-8
import os
import sys
from datetime import datetime

import numpy as np
import cv2
import tensorflow as tf

from tensorboard.plugins import projector
from prepare_data.utils import readlines
from train_models.MTCNN_config import config
from prepare_data.read_tfrecords import read_multi_tfrecords, read_single_tfrecord
from train_models.mtcnn_model import P_Net


def train_model(base_lr, loss, data_num):
    """
    train model
    :param base_lr: base learning rate
    :param loss: loss
    :param data_num:
    :return:
    train_op, lr_op
    """
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    # LR_EPOCH [8,14]
    # boundaried [num_batch,num_batch]
    boundaries = [int(epoch * data_num / config.BATCH_SIZE) for epoch in config.LR_EPOCH]
    # lr_values[0.01,0.001,0.0001,0.00001]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(config.LR_EPOCH) + 1)]
    # control learning rate
    lr_op = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.MomentumOptimizer(lr_op, 0.9)
    train_op = optimizer.minimize(loss, global_step)
    return train_op, lr_op


'''
certain samples mirror
def random_flip_images(image_batch,label_batch,landmark_batch):
    num_images = image_batch.shape[0]
    random_number = npr.choice([0,1],num_images,replace=True)
    #the index of image needed to flip
    indexes = np.where(random_number>0)[0]
    fliplandmarkindexes = np.where(label_batch[indexes]==-2)[0]
    
    #random flip    
    for i in indexes:
        cv2.flip(image_batch[i],1,image_batch[i])
    #pay attention: flip landmark    
    for i in fliplandmarkindexes:
        landmark_ = landmark_batch[i].reshape((-1,2))
        landmark_ = np.asarray([(1-x, y) for (x, y) in landmark_])
        landmark_[[0, 1]] = landmark_[[1, 0]]#left eye<->right eye
        landmark_[[3, 4]] = landmark_[[4, 3]]#left mouth<->right mouth        
        landmark_batch[i] = landmark_.ravel()
    return image_batch,landmark_batch
'''


# all mini-batch mirror
def random_flip_images(image_batch, label_batch, landmark_batch):
    # mirror
    if np.random.choice([0, 1]) > 0:
        num_images = image_batch.shape[0]
        fliplandmarkindexes = np.where(label_batch == -2)[0]
        flipposindexes = np.where(label_batch == 1)[0]
        # only flip
        flipindexes = np.concatenate((fliplandmarkindexes, flipposindexes))
        # random flip
        for i in flipindexes:
            cv2.flip(image_batch[i], 1, image_batch[i])

            # pay attention: flip landmark
        for i in fliplandmarkindexes:
            landmark_ = landmark_batch[i].reshape((-1, 2))
            landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
            landmark_[[0, 1]] = landmark_[[1, 0]]  # left eye<->right eye
            landmark_[[3, 4]] = landmark_[[4, 3]]  # left mouth<->right mouth
            landmark_batch[i] = landmark_.ravel()

    return image_batch, landmark_batch


def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs, max_delta=0.2)
    inputs = tf.image.random_saturation(inputs, lower=0.5, upper=1.5)

    return inputs


def train_pnet(input, prefix, end_epoch, base_dir, display=100, base_lr=0.01, seed=None):
    """
    train PNet/RNet/ONet
    :param net_factory:
    :param input: model path
    :param end_epoch:
    :param dataset:
    :param display:
    :param base_lr:
    :return:
    """
    np.random.seed(seed=seed)
    net_factory = P_Net
    type = 'PNet'

    label_file = base_dir.joinpath('train_landmark.txt')

    num = len(readlines(label_file))
    print("Total size of the dataset is: ", num)
    print(input)

    # PNet use this method to get data
    dataset_dir = input.joinpath('train_PNet_landmark.tfrecord')
    print('dataset dir is:', dataset_dir)

    tfdata = read_single_tfrecord(dataset_dir, config.BATCH_SIZE, type)

    # landmark_dir
    image_size = 12
    radio_cls_loss = 1.0
    radio_bbox_loss = 0.5
    radio_landmark_loss = 0.5

    # define placeholder
    input_image = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE, 10], name='landmark_target')

    # get loss and accuracy
    input_image = image_color_distort(input_image)
    cls_loss_op, bbox_loss_op, landmark_loss_op, l2_loss_op, accuracy_op = net_factory(input_image, label, bbox_target,
                                                                                       landmark_target, training=True)
    # train,update learning rate(3 loss)
    total_loss_op = radio_cls_loss * cls_loss_op + radio_bbox_loss * bbox_loss_op + radio_landmark_loss * landmark_loss_op + l2_loss_op
    train_op, lr_op = train_model(base_lr, total_loss_op, num)

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()

    # save model
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)

    # visualize some variables
    tf.summary.scalar("cls_loss", cls_loss_op)
    tf.summary.scalar("bbox_loss", bbox_loss_op)
    tf.summary.scalar("landmark_loss", landmark_loss_op)
    tf.summary.scalar("cls_accuracy", accuracy_op)
    tf.summary.scalar("total_loss", total_loss_op)
    summary_op = tf.summary.merge_all()

    logdir = prefix.parent.joinpath('logs')
    if not logdir.exists():
        logdir.mkdir(parents=True)

    writer = tf.summary.FileWriter(logdir.as_posix(), sess.graph)
    projector_config = projector.ProjectorConfig()
    projector.visualize_embeddings(writer, projector_config)
    # begin
    coord = tf.train.Coordinator()
    # begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    i = 0
    # total steps
    MAX_STEP = int(num / config.BATCH_SIZE + 1) * end_epoch
    epoch = 0
    sess.graph.finalize()

    try:
        for step in range(MAX_STEP):
            i = i + 1
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array, landmark_batch_array = sess.run(tfdata)

            # random flip
            image_batch_array, landmark_batch_array = random_flip_images(image_batch_array,
                                                                         label_batch_array,
                                                                         landmark_batch_array)
            '''
            print('im here')
            print(image_batch_array.shape)
            print(label_batch_array.shape)
            print(bbox_batch_array.shape)
            print(landmark_batch_array.shape)
            print(label_batch_array[0])
            print(bbox_batch_array[0])
            print(landmark_batch_array[0])
            '''

            _, _, summary = sess.run([train_op, lr_op, summary_op],
                                     feed_dict={input_image: image_batch_array,
                                                label: label_batch_array,
                                                bbox_target: bbox_batch_array,
                                                landmark_target: landmark_batch_array})

            if (step + 1) % display == 0:
                # acc = accuracy(cls_pred, labels_batch)
                cls_loss, bbox_loss, landmark_loss, L2_loss, lr, acc = sess.run(
                    [cls_loss_op, bbox_loss_op, landmark_loss_op, l2_loss_op, lr_op, accuracy_op],
                    feed_dict={input_image: image_batch_array, label: label_batch_array, bbox_target: bbox_batch_array,
                               landmark_target: landmark_batch_array})

                total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * bbox_loss + radio_landmark_loss * landmark_loss + L2_loss
                # landmark loss: %4f,
                print(
                    "%s : Step: %d/%d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,Landmark loss :%4f,L2 loss: %4f, Total Loss: %4f ,lr:%f " % (
                        datetime.now(), step + 1, MAX_STEP, acc, cls_loss, bbox_loss, landmark_loss, L2_loss,
                        total_loss, lr))

            # save every two epochs
            if i * config.BATCH_SIZE > 2*num:
                epoch = epoch + 1
                i = 0
                path_prefix = saver.save(sess, str(prefix), global_step=2 * epoch)
                print('path prefix is :', path_prefix)
            writer.add_summary(summary, global_step=step)
    except tf.errors.OutOfRangeError:
        pass
    finally:
        coord.request_stop()
        writer.close()

    coord.join(threads)
    sess.close()
