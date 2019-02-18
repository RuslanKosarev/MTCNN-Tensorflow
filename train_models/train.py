# coding:utf-8

import numpy as np
import cv2
import tensorflow as tf
from datetime import datetime
from tensorboard.plugins import projector
from prepare_data.read_tfrecords import read_multi_tfrecords


def train_model(base_lr, loss, iterations):
    """

    :param base_lr:
    :param loss:
    :param iterations:
    :return:
    """
    from train_models.MTCNN_config import config
    lr_factor = 0.1
    global_step = tf.Variable(0, trainable=False)
    # LR_EPOCH [8,14]
    # boundaried [num_batch,num_batch]
    boundaries = [int(epoch * iterations) for epoch in config.LR_EPOCH]
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


def train(config, tfrecord, prefix, display=100, seed=None):
    """

    :param config:
    :param tfrecord:
    :param prefix:
    :param display:
    :param seed:
    :return:
    """
    np.random.seed(seed=seed)

    if not prefix.parent.exists():
        prefix.parent.mkdir(parent=True)

    logdir = prefix.parent.joinpath('logs')
    if not logdir.exists():
        logdir.mkdir()

    image_size = config.image_size
    radio_cls_loss = 1.0
    radio_bbox_loss = 0.5
    radio_landmark_loss = 0.5

    batch_size = config.batch_size

    pos_batch_size = int(batch_size*config.pos_ratio)
    part_batch_size = int(batch_size*config.part_ratio)
    neg_batch_size = int(batch_size*config.neg_ratio)
    landmark_batch_size = int(batch_size*config.landmark_ratio)

    batch_sizes = [pos_batch_size, part_batch_size, neg_batch_size, landmark_batch_size]
    batch_size = sum(batch_sizes)
    tfdata = read_multi_tfrecords(config, tfrecord, batch_sizes)

    # define placeholder
    input_image = tf.placeholder(tf.float32, shape=[batch_size, image_size, image_size, 3], name='input_image')
    label = tf.placeholder(tf.float32, shape=[batch_size], name='label')
    bbox_target = tf.placeholder(tf.float32, shape=[batch_size, 4], name='bbox_target')
    landmark_target = tf.placeholder(tf.float32, shape=[batch_size, 10], name='landmark_target')

    input_image = image_color_distort(input_image)

    net = config.factory(input_image, label, bbox_target, landmark_target, training=True)

    # initialize total loss
    total_loss = radio_cls_loss * net.cls_loss + radio_bbox_loss * net.bbox_loss + radio_landmark_loss * net.landmark_loss + net.l2_loss
    train_op, lr_op = train_model(config.lr, total_loss, config.number_of_iterations)

    init = tf.global_variables_initializer()
    sess = tf.Session()

    # save model
    saver = tf.train.Saver(max_to_keep=0)
    sess.run(init)

    # visualize some variables
    tf.summary.scalar('cls_loss', net.cls_loss)
    tf.summary.scalar('bbox_loss', net.bbox_loss)
    tf.summary.scalar('landmark_loss', net.landmark_loss)
    tf.summary.scalar('cls_accuracy', net.accuracy)
    tf.summary.scalar('total_loss', total_loss)
    summary_op = tf.summary.merge_all()

    writer = tf.summary.FileWriter(str(logdir), sess.graph)
    projector_config = projector.ProjectorConfig()
    projector.visualize_embeddings(writer, projector_config)
    # begin
    coord = tf.train.Coordinator()
    # begin enqueue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.graph.finalize()

    # total steps
    number_of_iterations = config.number_of_iterations * config.number_of_epochs

    try:
        for it in range(number_of_iterations):
            if coord.should_stop():
                break
            image_batch_array, label_batch_array, bbox_batch_array, landmark_batch_array = sess.run(tfdata)

            # random flip
            image_batch_array, landmark_batch_array = random_flip_images(image_batch_array,
                                                                         label_batch_array,
                                                                         landmark_batch_array)

            _, _, summary = sess.run([train_op, lr_op, summary_op], feed_dict={input_image: image_batch_array,
                                                                               label: label_batch_array,
                                                                               bbox_target: bbox_batch_array,
                                                                               landmark_target: landmark_batch_array})
            final = (it+1) == number_of_iterations

            if (it+1) % display == 0 or final:
                fetches = (net.cls_loss, net.bbox_loss, net.landmark_loss, net.l2_loss, net.accuracy, total_loss, lr_op)
                values = sess.run(fetches, feed_dict={input_image: image_batch_array,
                                                      label: label_batch_array,
                                                      bbox_target: bbox_batch_array,
                                                      landmark_target: landmark_batch_array})

                names = ('cls loss', 'bbox loss', 'landmark loss', 'l2 loss', 'accuracy', 'total loss', 'lr')
                info = '{}: iteration: {}/{}'.format(datetime.now(), it + 1, number_of_iterations)
                for name, value in zip(names, values):
                    info += ', {}: {:0.5f}'.format(name, value)

                print(info)

            # save every step
            if (it+1) % (number_of_iterations / config.number_of_epochs) == 0 or final:
                epoch = int(config.number_of_epochs * (it + 1) / number_of_iterations)
                path_prefix = saver.save(sess, str(prefix), global_step=epoch)
                print('path prefix is:', path_prefix, 'epoch', epoch, '/', config.number_of_epochs)
                writer.add_summary(summary, global_step=it)

    except tf.errors.OutOfRangeError:
        pass
    finally:
        coord.request_stop()
        writer.close()

    coord.join(threads)
    sess.close()
