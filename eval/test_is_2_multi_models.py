#coding=utf8
# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys
import fnmatch

import torchvision

MODEL_DIR = './imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
    #assert(type(images) == list)
    #print (type(images[0]))
    #assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = 100
    with tf.Session() as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in range(n_batches):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {'ExpandDims:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
          part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
          kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
          kl = np.mean(np.sum(kl, 1))
          scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception():
    global softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    print (filepath)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        print(DATA_URL)
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    # Works with an arbitrary minibatch size.
    with tf.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o._shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)

if softmax is None:
  _init_inception()

def load_data(fullpath):
    print(fullpath)
    images = []
    for path, subdirs, files in os.walk(fullpath):
        for name in files:
            if name.rfind('.jpg') != -1 or name.rfind('.png') != -1:
                filename = os.path.join(path, name)
                # print('filename', filename)
                # print('path', path, '\nname', name)
                # print('filename', filename)
                if os.path.isfile(filename):
                    img = scipy.misc.imread(filename)
                    images.append(img)
    print('images', len(images), images[0].shape)
    return images

def load_data_path_list(path_list):
    images = []
    for filename in path_list:
        if os.path.isfile(filename):
            img = scipy.misc.imread(filename)
            images.append(img)

    print(len(images))
    print (len(path_list))
    print('images', len(images), images[0].shape)
    return images

def get_IS(path_list):
    images = load_data_path_list(path_list)
    mean, std = get_inception_score(images)

    return mean, std

def get_fakeimg_paths(path):
    paths1, paths2, paths3, paths4, paths5, paths6, paths7 = [],[],[],[],[],[],[]
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*_full_fake.jpg'):
            paths1.append(os.path.join(path, filename))

    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*_noGAN_fake.jpg'):
            paths2.append(os.path.join(path, filename))

    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*_noPH_fake.jpg'):
            paths3.append(os.path.join(path, filename))

    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*_noVGG_fake.jpg'):
            paths4.append(os.path.join(path, filename))

    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*_noL1_fake.jpg'):
            paths5.append(os.path.join(path, filename))

    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*_woWB_fake.jpg'):
            paths6.append(os.path.join(path, filename))

    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*_woStageI_fake.jpg'):
            paths7.append(os.path.join(path, filename))




    return paths1, paths2, paths3, paths4, paths5, paths6, paths7

if __name__ == "__main__":
    # trainset = torchvision.datasets.CIFAR10(root='./', download=True)
    # x = trainset.train_data
    # mean, std = get_inception_score(x)
    # print("score = {} +- {}".format(mean, std))

    # score = 11.23736763 + - 0.116222046316 batch_size 100
    # score = 11.23736763 + - 0.116222389042  batch_size 32
    # score = 11.23736763 +- 0.116221621633  batch_size 20
    # score = 11.23736763 + - 0.116222389042 batch_size 10

    ###===========================================
    if len(sys.argv) == 1:
        image_dir = '../results/mGPU_nofusion_noD2D3_lightCNN_tv_corr_sia_block3_bz6_0115/test_latest/images'  ### 51111号
    elif len(sys.argv) == 2:
        image_dir = sys.argv[1]
    elif len(sys.argv) == 3:
        model_name = sys.argv[1]
        epoch  = sys.argv[2]
        image_dir = '../results/' + model_name + '/' + epoch + '/images'

    print(image_dir)
    paths1, paths2, paths3, paths4, paths5, paths6, paths7 = get_fakeimg_paths(image_dir)

    mean1, std1 = get_IS(paths1)
    mean2, std2 = get_IS(paths2)
    mean3, std3 = get_IS(paths3)
    mean4, std4 = get_IS(paths4)
    mean5, std5 = get_IS(paths5)
    mean6, std6 = get_IS(paths6)
    mean7, std7 = get_IS(paths7)
    print("full_fake IS score = {} +- {}".format(mean1, std1))
    print("noGAN_fake IS score = {} +- {}".format(mean2, std2))
    print("noPH_fake IS score = {} +- {}".format(mean3, std3))
    print("noVGG_fake IS score = {} +- {}".format(mean4, std4))
    print("noL1_fake IS score = {} +- {}".format(mean5, std5))
    print("woWB_fake IS score = {} +- {}".format(mean6, std6))
    print("woStageI_fake IS score = {} +- {}".format(mean7, std7))




