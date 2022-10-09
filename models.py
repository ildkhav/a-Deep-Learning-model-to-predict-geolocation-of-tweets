"""
TensorFlow CNN models.

These are predefined Conv1D models with some hardcoded parameters, other part of which may be tuned on instantiating.
Classification and regression models defined with single and multi Conv1D layers each.
models_factory() function accepts predefined names and return corresponding model instance.
"""
import time
from typing import List
from typing import Optional
from typing import Any

from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers import Dense
from keras.layers import Softmax
import math
import matplotlib.pyplot as plt
import tensorflow as tf


def mean_error_km(y_true: tf.Tensor, y_predicted: tf.Tensor) -> float:
    """
    Given sets of coordinates (true and predicted) calculate distance between corresponding pairs,
    then find mean value of the distances and return it.
    Distance calculation is based on Haversine formula.
    :param y_true:
    :param y_predicted:
    :return:
    """
    m = tf.constant([180, 360], dtype=tf.float32)
    s = tf.constant([90, 180], dtype=tf.float32)

    y_true = y_true * m - s
    y_predicted = y_predicted * m - s

    y_true = y_true * math.pi / 180
    y_predicted = y_predicted * math.pi / 180

    y_true = tf.transpose(y_true)
    y_predicted = tf.transpose(y_predicted)

    delta_lat = y_predicted[0] - y_true[0]
    delta_lon = y_predicted[1] - y_true[1]

    m = tf.math.sin(delta_lat / 2) ** 2 + \
        tf.math.cos(y_true[0]) * tf.math.cos(y_predicted[0]) * tf.math.sin(delta_lon / 2) ** 2

    return tf.reduce_mean(2 * 6371 * tf.math.asin(tf.math.sqrt(m)))


class BasicModel:
    def __init__(self, name: str, model: Sequential):
        """
        Basic class, provides fit, save methods, which simply call eponymous Keras methods.
        :param name: model's name.
        :param model: Keras Sequential model instance.
        """
        self.name = name
        self.model = model

    def fit(self, x: tf.Tensor, y: tf.Tensor, **kwargs) -> tf.keras.callbacks.History:
        """
        Call Keras fit method.
        :param x:
        :param y:
        :param kwargs: parameters to pass to Keras fit method.
        :return: tf.keras.callbacks.History
        """
        return self.model.fit(x, y, **kwargs)

    def save(self, path: str, **kwargs) -> Optional[Any]:
        """
        Save the model by given path, it calls Keras save method.
        :param path: where to save the model.
        :param kwargs: optional arguments to Keras save method.
        :return: Optional[Any]
        """
        return self.model.save(path, **kwargs)


class RegressionModel(BasicModel):
    def __init__(self, name: str, model):
        """
        Regression model basic class, extends BasicModel, adds plot method to make charts.
        :param name: model's name
        :param model: model's instance
        """
        super().__init__(name, model)

    def plot(self, path: str, history: tf.keras.callbacks.History) -> List[str]:
        """
        Plot history while training, save charts into given directory, return the charts full path file names.
        :param path: a directory where to save charts.
        :param history:
        :return: tf.keras.callbacks.History object
        """
        path = path.rstrip('/')

        plt.clf()
        plt.plot(history.history['val_root_mean_squared_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title("Validation error, normalized degrees")
        plt.ylabel('error value')
        plt.xlabel('epoch')
        plt.legend(['RMSE', 'MAE'], loc='upper left')
        plt.tight_layout()
        degrees_chart = f'{path}/{self.name}_validation_error_degrees_{int(time.time())}.png'
        plt.savefig(degrees_chart)

        # plot MAE in km
        plt.clf()
        plt.plot(history.history['val_mean_error_km'])
        plt.title("Validation error in km")
        plt.ylabel('km')
        plt.xlabel('epoch')
        plt.tight_layout()
        km_chart = f'{path}/{self.name}_validation_error_km_{int(time.time())}.png'
        plt.savefig(km_chart)

        return [degrees_chart, km_chart]


class ClassificationModel(BasicModel):
    def __init__(self, name: str, model: Sequential):
        """
        Classification model basic class, extends BasicModel, adds plot method to make charts.
        :param name: model's name
        :param model: model's instance
        """
        super().__init__(name, model)

    def plot(self, path: str, history: tf.keras.callbacks.History) -> str:
        """
        Plot history while training, save chart into given directory, return the chart full path file names.
        :param path: a directory where to save charts.
        :param history: tf.keras.callbacks.History object.
        :return: saved chart full path file name
        """
        path = path.rstrip('/')

        plt.clf()
        plt.plot(history.history['val_accuracy'])
        plt.title("The model's accuracy")
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.tight_layout()
        accuracy_chart = f'{path}/{self.name}_accuracy_{int(time.time())}.png'
        plt.savefig(accuracy_chart)

        return accuracy_chart


class SingleLayerRegressionModel(RegressionModel):
    def __init__(self, **kwargs):
        """
        Single Conv1D layer regression model class, inherits fit, save, plot methods from RegressionModel.
        Needs Keras Conv1D filters, input_shape parameters to be provided. Output dimension is of shape (None, 1, 2).
        :param kwargs: Conv1D layer filters, input_shape parameters are required.
        """
        name = 'single_layer_regression'
        model = Sequential(name=name)
        # Dropout layer may be added so as not to over fit, better off to test
        # model.add(Dropout(.05, input_shape=(1, twl)))
        model.add(Conv1D(filters=kwargs['filters'],
                         kernel_size=15,
                         strides=1,
                         padding='same',
                         activation='relu',
                         input_shape=(1, kwargs['max_tweet_length'])))

        model.add(Dense(2, activation='relu'))

        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.RootMeanSquaredError(),
                               tf.keras.metrics.MeanAbsoluteError(),
                               mean_error_km])

        super().__init__(name, model)


class MultiLayerRegressionModel(RegressionModel):
    def __init__(self, **kwargs):
        """
        4 Conv1D layers regression model class, inherits fit, save, plot methods from RegressionModel.
        Needs Keras Conv1D filters, input_shape parameters to be provided. Output dimension is of shape (None, 1, 2).
        :param kwargs: first Conv1D layer filters, input_shape parameters are required.
        """
        name = 'multiple_layers_regression'
        model = Sequential(name=name)
        # Dropout layer may be added so as not to over fit, better off to test
        # model.add(Dropout(.05, input_shape=(1, kwargs['max_tweet_length'])))
        model.add(
            Conv1D(filters=kwargs['filters'],
                   kernel_size=3,
                   strides=1,
                   padding='same',
                   activation='relu',
                   input_shape=(1, kwargs['max_tweet_length'])))
        model.add(Conv1D(filters=kwargs['filters'], kernel_size=2, strides=1, padding='same', activation='relu'))
        model.add(Conv1D(filters=kwargs['filters'], kernel_size=2, strides=1, padding='same', activation='relu'))
        model.add(Conv1D(filters=kwargs['filters'], kernel_size=2, strides=1, padding='same', activation='relu'))
        model.add(Dense(2, activation='relu'))

        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.metrics.RootMeanSquaredError(),
                               tf.keras.metrics.MeanAbsoluteError(),
                               mean_error_km])
        super().__init__(name, model)


class SingleLayerClassification(ClassificationModel):
    def __init__(self, **kwargs):
        """
        Single Conv1D layer classification model, inherits fit, save, plot methods from ClassificationModel.
        Needs Keras Conv1D filters, input_shape parameters to be provided.
        Output shape dimension depends on required precision parameter, the bigger precision value (in degrees)
        the fewer factors will be. Output shape in general will be like (None, 1, <factors length>).
        :param kwargs: Conv1D layer filters, input_shape parameters are required. A precision parameters which shapes
        output Dense layer is required.
        """
        name = 'single_layer_classification'
        model = Sequential(name=name)
        # Dropout layer may be added so as not to over fit, better off to test
        # model.add(Dropout(.05, input_shape=(1, kwargs['max_tweet_length'])))
        model.add(
            Conv1D(filters=kwargs['filters'],
                   kernel_size=15,
                   strides=1,
                   padding='same',
                   activation='relu',
                   input_shape=(1, kwargs['max_tweet_length'])))
        model.add(Dense((180 / kwargs['precision'] + 1) * (360 / kwargs['precision'] + 1), activation='relu'))
        model.add(Softmax())

        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        super().__init__(name, model)


class MultiLayerClassification(ClassificationModel):
    def __init__(self, **kwargs):
        """
        4 Conv1D layer classification model, inherits fit, save, plot methods from ClassificationModel.
        Needs Keras Conv1D filters, input_shape parameters to be provided.
        Output shape dimension depends on required precision parameter, the bigger precision value (in degrees)
        the fewer factors will be. Output shape in general will be like (None, 1, <factors length>).
        :param kwargs: first Conv1D layer filters, input_shape parameters are required. A precision parameters which
        shapes output Dense layer is required.
        """
        name = 'multiple_layers_classification'
        model = Sequential(name=name)
        # Dropout layer may be added so as not to over fit, better off to test
        # model.add(Dropout(.05, input_shape=(1, twl)))
        model.add(
            Conv1D(filters=kwargs['filters'],
                   kernel_size=3,
                   strides=1,
                   padding='same',
                   activation='relu',
                   input_shape=(1, kwargs['max_tweet_length'])))
        model.add(Conv1D(filters=kwargs['filters'], kernel_size=2, strides=1, padding='same', activation='relu'))
        model.add(Conv1D(filters=kwargs['filters'], kernel_size=2, strides=1, padding='same', activation='relu'))
        model.add(Conv1D(filters=kwargs['filters'], kernel_size=2, strides=1, padding='same', activation='relu'))
        model.add(Dense((180 / kwargs['precision'] + 1) * (360 / kwargs['precision'] + 1), activation='relu'))
        model.add(Softmax())

        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        super().__init__(name, model)


def models_factory(**kwargs) -> Sequential:
    """
    Return corresponding class instance given a name. The name is composed of kwargs keys layers_number (single|multi),
    model_type (regression|classification). The kwargs are also passed to the class instance being instantiated.
    :param kwargs: model's type
    :return: Sequential
    """
    models = {'single_regression': SingleLayerRegressionModel,
              'multi_regression': MultiLayerRegressionModel,
              'single_classification': SingleLayerClassification,
              'multi_classification': MultiLayerClassification}

    return models[f'{kwargs["layers_number"]}_{kwargs["model_type"]}'](**kwargs)
