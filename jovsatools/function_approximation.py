# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/function_approximation.ipynb (unless otherwise specified).

__all__ = ['KerasPipeline', 'HighLevelKerasPipeline', 'HighLevelKerasPipelineMultiTarget']

# Cell
from abc import ABC, abstractmethod
from fastcore.test import *
import jovsatools
from jovsatools import data_generator
from nbdev.showdoc import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

test_eq(tf.__version__>= "2.2.0", True)

# Cell

class KerasPipeline(ABC):
    """Scafolding for a tf.keras model building pipeline.

    This base class will contain scafolding for all pipelines
    used. The goal of this base class is mainly to add safeguard
    and assumption enforcement.
    """
    def __init__(self):
        # safe guard: to ensure that various session don't collide
        keras.backend.clear_session()
        self.train_x, self.train_y, self.test_x, self.test_y = None, None, None, None
        self.model = None
        self.evaluation = None
        self.train_history = None

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def run_training(self):
        pass

    @abstractmethod
    def evaluate_pipeline(self):
        pass

    def plot_model_shapes(self, show_shapes=True, show_layer_names=True):
        return keras.utils.plot_model(self.model,
                                      show_shapes=show_shapes,
                                      show_layer_names=show_layer_names)

    def keras_model_summary(self):
        return self.model.summary()

    def __call__(self, verbose=0):
        ########################################
        # step 1: get data
        ########################################
        self.get_data()
        # checking that self.get_data() is implimented as expected
        assert list(
            map(lambda x: x is not None, [self.train_x, self.train_y, self.test_x, self.test_y])
            )  == [True] * 4

        ########################################
        # step 2: build model
        ########################################
        self.build_model()
        # checking that self.model() is implimented as expected
        assert self.model is not None

        ########################################
        # step 3: train model
        ########################################
        print("\n", "=-="*10, "MODEL TRAINING", "=-="*10, "\n")
        self.run_training(verbose)
        # checking that self.train() is implimented as expected
        assert self.train_history is not None
        assert self.model._is_compiled == True

        if verbose > 0:
            print("\n", "=-="*10, "MODEL SUMMARY", "=-="*10, "\n")
            self.keras_model_summary()

        ########################################
        # step 4: evaluate pipeline
        ########################################
        print("\n", "=-="*10, "PIPELINE EVALUATION", "=-="*10, "\n")
        self.evaluate_pipeline()
        # checking that self.evaluate() is implimented as expected
        assert self.evaluation is not None

        # TODO (jovsa): run tests that ensure pipeline trains model above a benchmark

# Cell
class HighLevelKerasPipeline(KerasPipeline):
    def __init__(self, sample_n, additional_y):
        self.sample_n = sample_n
        self.additional_y = additional_y
        super().__init__()

    def get_data(self):
        data = data_generator.MNISTDataGenerator(self.additional_y)
        datasets  = data(self.sample_n)
        self.train_x, self.train_y, self.test_x, self.test_y = datasets

    def build_model(self):
        inputs = keras.Input(shape=(784, 1), name="inputs")
        # Rescale images to [0, 1]
        x = Rescaling(scale=1./255)(inputs)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=64, activation='relu')(x)
        output = keras.layers.Dense(units=10, activation='softmax', name="output")(x)

        self.model = keras.Model(inputs=inputs, outputs=output)

    def run_training(self, verbose):
        self.model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()])

        self.train_history = self.model.fit(
            x=self.train_x, y=self.train_y, epochs=10, batch_size=64,
            verbose=verbose, validation_split=0.1)

    def evaluate_pipeline(self):
        self.evaluation = self.model.evaluate(x=self.test_x, y=self.test_y)

# Cell
class HighLevelKerasPipelineMultiTarget(KerasPipeline):
    def __init__(self, sample_n, additional_y):
        self.sample_n = sample_n
        self.additional_y = additional_y
        super().__init__()

    def get_data(self):
        data = data_generator.MNISTDataGenerator(self.additional_y)
        datasets  = data(self.sample_n)
        self.train_x, self.train_y, self.test_x, self.test_y = datasets


    def build_model(self):
        inputs = keras.Input(shape=(784, 1), name="inputs")
        # Rescale images to [0, 1]
        x = Rescaling(scale=1./255)(inputs)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=64, activation='relu')(x)
        output_1 = keras.layers.Dense(units=10, activation='softmax', name="output_0")(x)
        output_2 = keras.layers.Dense(units=1, activation='sigmoid', name="output_1")(x)

        self.model = keras.Model(inputs=inputs, outputs=[output_1, output_2])

    def custom_loss_0(self):
        def loss_fn(y_true, y_pred):
            l = keras.losses.SparseCategoricalCrossentropy()
            return l(y_true, y_pred)
        return loss_fn

    def custom_loss_1(self):
        def loss_fn(y_true, y_pred):
            l = keras.losses.CategoricalCrossentropy()
            return l(y_true, y_pred)
        return loss_fn


    def run_training(self, verbose):

        self.model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
            loss = {
                "output_0": self.custom_loss_0(),
                "output_1": self.custom_loss_1(),
            },
            metrics={
                "output_0": keras.metrics.SparseCategoricalAccuracy(),
                "output_1": "categorical_accuracy"
            })

        self.train_history = self.model.fit(
            x=self.train_x,
            y={
                "output_0": self.train_y[:,0],
                "output_1": self.train_y[:,1]
            },
            epochs=30, batch_size=64,
            verbose=verbose, validation_split=0.1)

    def evaluate_pipeline(self):
        self.evaluation = self.model.evaluate(x=self.test_x, y={"output_0": self.test_y[:,0], "output_1": self.test_y[:,1]})