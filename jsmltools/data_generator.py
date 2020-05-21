# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/data_generator.ipynb (unless otherwise specified).

__all__ = ['MNISTDataGenerator']

# Cell
from fastcore.test import *
from nbdev.showdoc import *
import numpy as np
import jsmltools
import tensorflow as tf

# Cell

class MNISTDataGenerator(object):
    """Generates a multi-calss regression/classification data based on MNIST

        x = mnist data flattened
        y = regression/classification targets based on self.initialize()

        Arguments:
            additional_y: Integer, used to specify the additional targets
            seed = Integer, used to specify np.random.seed()

        Returns:
            train_x: numpy.ndarray, MNIST train data flattened
            train_y: numpy.ndarray, all the targets; MNIST label + additional_y
            test_x: numpy.ndarray, MNIST test data flattened
            test_y: numpy.ndarray, all the targets; MNIST label + additional_y
    """
    def __init__(self, additional_y=0, seed=1123):
        self.additional_y = additional_y
        self.seed = np.random.seed(seed=seed)

        # needed to store additional generation functions
        self._func_map = {}
        self._initialize(additional_y)

        assert list(filter(lambda x: x is None, [
            self.mnist_train_x,
            self.mnist_train_y,
            self.mnist_test_x,
            self.mnist_test_y
        ])) == []

        self.train_n = len(self.mnist_train_x)
        self.test_n = len(self.mnist_test_x)

    def _initialize(self, additional_y):
        """Prepare functions to approximate """
        epsilon = 0.000123
        C = 102 # emperical value from analyzing MNIST data

        # label 0: mnist class
        if additional_y >= 1:
            # label 1: classification (0, 1) target
            self._func_map[0] = lambda x: int(np.random.random()<0.9) if 2*(x[10]**3)-2*x[3]+15 > 8.9 else 0

        if additional_y >= 2:
            # label 2: unbounded regression target
            self._func_map[1] = lambda x: np.log(np.sum(x)+epsilon)

        if additional_y >= 3:
            # label 3: bounded (0, 1) regression target
            self._func_map[2] = lambda x: np.mean(x)/C

        # regular MNIST train/test data
        train, test = tf.keras.datasets.mnist.load_data()
        self.mnist_train_x, self.mnist_train_y = train[0], train[1]
        self.mnist_test_x, self.mnist_test_y = test[0], test[1]

    def prepare_datasets(self, mnist_x, mnist_y, n):
        """Main worker function for dataset preparation """
        x, y = [], []
        for i in range(n):
            features = mnist_x[i].flatten().reshape(-1, 1)
            labels = [mnist_y[i]]
            for f in range(self.additional_y):
                labels.append(self._func_map[f](features))
            x.append(features)
            y.append(labels)
        return np.asarray(x), np.asarray(y)

    def __call__(self, sample_n):
        """Data generation call """
        assert sample_n <= self.train_n, f"max alloable size is {self.train_n}"
        train_x, train_y = [], []
        test_x, test_y = [], []

        train_x, train_y = self.prepare_datasets(self.mnist_train_x, self.mnist_train_y, n=sample_n)
        test_x, test_y = self.prepare_datasets(self.mnist_test_x, self.mnist_test_y, n=self.test_n)
        return (train_x, train_y, test_x, test_y)