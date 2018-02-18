# adapted from tensorflow's tensorflow.contrib.learn.python.learn.datasets.mnist file
# and from keras dataset loading files
# very little of this code is originally written, and all credit should go to the original authors
# the relevant original code has been brought here to help limit dependency and package compatibility issues
import os
import sys
import gzip
import pickle
import numpy as np
from scipy.io import loadmat
from collections import namedtuple

from data_utils import get_file

DATASETS_AVAILABLE = ['mnist', 'frey_face', 'fashion_mnist']
# DATASETS_AVAILABLE = ['mnist', 'frey_face', 'fashion_mnist', 'cifar10', 'cifar100']  # TODO: color image datasets
Datasets = namedtuple('Datasets', ['train', 'validation', 'test'])


class Dataset:
    def __init__(self,
                 images,
                 labels,
                 img_dims,
                 dtype=np.float32,
                 reshape=True,
                 seed=123):
        np.random.seed(seed)  # set seed elsewhere?
        if dtype not in (np.uint8, np.float32):
            # dtype should be either uint8 to leave input as [0, 255] or float32 to rescale into [0.0, 1.0]
            raise TypeError(
                'Invalid image dtype {}, expected uint8 or float32'.format(dtype))
        assert images.shape[0] == labels.shape[0], (
            'images.shape: {} labels.shape: {}'.format(images.shape, labels.shape))
        assert type(seed) is int, (
            'Invalid seed specified: {}'.format(seed))
        self._num_examples = images.shape[0]

        # flatten images
        # TODO: adjust for color images
        if reshape:
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        if dtype == np.float32:
            # convert from [0, 255] --> [0.0, 1.0]
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._img_dims = img_dims
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._cur_epoch_completed = False

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_dims(self):
        return self._img_dims

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def cur_epoch_completed(self):
        return self._cur_epoch_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        self._cur_epoch_completed = False

        # shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]

        if start + batch_size > self._num_examples:
            # finished epoch
            self._epochs_completed += 1
            self._cur_epoch_completed = True

            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]

            # start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]

            return np.concatenate(
                (images_rest_part, images_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self.labels[start:end]


def load_data(dataset='mnist', dtype=np.float32, reshape=True, seed=123):
    # more clever way of handling this?
    if dataset == 'mnist':
        # datasets provided as tuples (imgs, labels) of data type uint8 (imgs in [0, 256])
        (train_images, train_labels), (test_images, test_labels) = _load_mnist()
    elif dataset == 'frey_face':
        # for now, returning an (n_imgs, 1) array of zeros for training labels
        (train_images, train_labels), (test_images, test_labels) = _load_freyface()  # unlabeled dataset
    elif dataset == 'fashion_mnist':
        (train_images, train_labels), (test_images, test_labels) = _load_fashion_mnist()
    # TODO: color images
    # loading functions for cifar10 and cifar100 work,
    #   but need to implement ability to accept color images upstream
    # elif dataset == 'cifar10':
    #     train_images, train_labels = _load_cifar10()
    # elif dataset == 'cifar100':
    #     train_images, train_labels = _load_cifar_100()
    # elif dataset == 'celebA':
    #     train_images, train_labels = _load_celebA()
    else:
        raise ValueError(
            'Unavailable dataset specified. Datasets available: [{}]'.format(', '.join(DATASETS_AVAILABLE)))

    # if not 0 <= validation_size <= train_images.shape[0]:
    #     raise ValueError('Validation size should be between 0 and {}. Received {}.'
    #                      .format(train_images.shape[0], validation_size))

    # no point in validation set here?
    # validation_images = train_images[:validation_size]
    # validation_labels = train_labels[:validation_size]
    # train_images = train_images[validation_size:]
    # train_labels = train_labels[validation_size:]

    img_dims = list(train_images.shape[1:])
    options = dict(img_dims=img_dims, dtype=dtype, reshape=reshape, seed=seed)

    train = Dataset(train_images, train_labels, **options)
    # validation = Dataset(validation_images, validation_labels, **options)
    test = Dataset(test_images, test_labels, **options)

    return Datasets(train=train, validation=None, test=test)


def _load_mnist(path='mnist.npz'):
    path = get_file(path,
                    origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
                    file_hash='8a61469f7ea1b51cbae51d4f78837e45')
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()

    # prevent compatibility issues
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    # unsupervised task
    # x_train = np.concatenate((x_train, x_test), axis=0)
    # y_train = np.concatenate((y_train, y_test), axis=0)

    return (x_train, y_train), (x_test, y_test)


def _load_freyface(path='frey_rawface.mat'):
    img_dims = [28, 20]
    path = get_file(path,
                    origin='https://cs.nyu.edu/~roweis/data/frey_rawface.mat')
    f = loadmat(path)
    x_train = f['ff']  # TODO: test set

    # reformat data to match expected format
    n_imgs = x_train.shape[1]
    x_train = x_train.transpose()
    x_train = np.reshape(x_train, tuple([n_imgs] + img_dims), order='C')
    x_train = np.expand_dims(x_train, axis=-1)

    # TODO: figure out better way of handling this
    return x_train, np.zeros(shape=(n_imgs, 1), dtype=np.uint8)


def _load_fashion_mnist():
    dirname = os.path.join('datasets', 'fashion-mnist')
    base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz']

    paths = []
    for fname in files:
        paths.append(get_file(fname,
                              origin=base + fname,
                              cache_subdir=dirname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8,
                                offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8,
                               offset=16).reshape(len(y_test), 28, 28)

    # prevent compatibility issues
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    # unsupervised task
    # x_train = np.concatenate((x_train, x_test), axis=0)
    # y_train = np.concatenate((y_train, y_test), axis=0)

    return (x_train, y_train), (x_test, y_test)


def _load_cifar10():
    dirname = 'cifar-10-batches-py'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = _load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = _load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # make channels last dimension
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    # unsupervised task
    # x_train = np.concatenate((x_train, x_test), axis=0)
    # y_train = np.concatenate((y_train, y_test), axis=0)

    return (x_train, y_train), (x_test, y_test)


def _load_cifar_100(label_mode='fine'):
    if label_mode not in ['fine', 'coarse']:
        raise ValueError('`label_mode` must be one of `"fine"`, `"coarse"`.')

    dirname = 'cifar-100-python'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    path = get_file(dirname, origin=origin, untar=True)

    fpath = os.path.join(path, 'train')
    x_train, y_train = _load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(path, 'test')
    x_test, y_test = _load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # make channels last dimension
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    # unsupervised task
    # x_train = np.concatenate((x_train, x_test), axis=0)
    # y_train = np.concatenate((y_train, y_test), axis=0)

    return (x_train, y_train), (x_test, y_test)


# def _load_celebA():
#     dirname = 'celebA-python'
#     origin = 'https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg/Img/img_align_celeba.zip'


def _load_batch(fpath, label_key='labels'):
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


# for debugging
if __name__ == "__main__":
    dataset = load_data(dataset='cifar100')
    print(dataset.train.num_examples)
