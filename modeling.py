"""
Convolutional Neural Network modeling.

Create classification, regression CNN models to predict and evaluate geolocation provided tweet text.
The models are being trained on data, which are csv files containing `text` column.
The csv file is expected to be UTF encoded, so it doesn't matter what the language is
or whether there are any emojis etc. The csv files are expected to be named as the following:
<latitude>_<longitude>.csv, that is the coordinates are encoded into the file name
and a given file contains only tweets for this coordinate.
On training the model data is split into train and test sets, ratio is provided via command line.
On evaluating, a test set is expected to be in the same format, that is csv, the `text` column, files naming convention.
On predicting, a tweet text to predict coordinates for is being provided via command line.
This utility employs Keras API and uses models.py module, where actual CNN models are located.
In this script one can find files processing logic and calling of models depending on command line arguments.
"""
import argparse
import glob
import logging
import math
import os
import re
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas
import tensorflow as tf
import pandas as pd

import models


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments. Create, predict, evaluate actions are different sub parsers.

    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description='Create a model, predict geolocation or '
                                                 'evaluate predicted location using existing model.')

    subparsers = parser.add_subparsers(help='Actions to perform', dest='subparser_name')
    #
    subparser_create = subparsers.add_parser('create', help='build a model')
    subparser_create.add_argument('--model-type',
                                  required=True,
                                  choices=['classification', 'regression'],
                                  help='A type of the model - classification or regression.')
    subparser_create.add_argument('--layers-number',
                                  required=True,
                                  choices=['single', 'multi'],
                                  help='A number of layers to be used to build the model, single layer of multi.')
    subparser_create.add_argument('--model-path', required=True, type=str, help='A path to save the model to.')
    subparser_create.add_argument('--max-tweet-length',
                                  required=False,
                                  type=int,
                                  default=80,
                                  help='Max tweet length allowed while building the model, '
                                       'longer than that will be cut.')
    subparser_create.add_argument('--data-path',
                                  required=True,
                                  type=str,
                                  help='A directory to get data.')
    subparser_create.add_argument('--test-fraction',
                                  required=False,
                                  type=int,
                                  default=10,
                                  choices=range(10, 40, 10),
                                  help='A fraction of data (per cent) to be used for testing.')
    subparser_create.add_argument('--precision',
                                  required=False,
                                  type=int,
                                  default=3,
                                  choices=range(1, 5),
                                  help='Geolocation precision in degrees for classification.')
    subparser_create.add_argument('--filters',
                                  required=False,
                                  type=int,
                                  default=1000,
                                  help='Number of CNN filters.')
    subparser_create.add_argument('--epochs',
                                  required=False,
                                  type=int,
                                  default=10,
                                  help='Number of epochs to train the model.')
    subparser_create.add_argument('--charts-directory',
                                  required=False,
                                  type=str,
                                  help='A directory to save charts.')

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--model-path', required=True, help='A path to the saved model.')

    subparser_predict = subparsers.add_parser('predict',
                                              help='Predict location given tweet text.',
                                              parents=[parent_parser])
    subparser_predict.add_argument('--tweet-text',
                                   required=True,
                                   type=str,
                                   help='A tweet text to predict coordinates for.')
    subparser_predict.add_argument('--charts-directory',
                                   required=False,
                                   type=str,
                                   help='A directory to save a chart for classification confidence distribution.')

    subparser_evaluate = subparsers.add_parser('evaluate', help='evaluate existing model',
                                               parents=[parent_parser])
    subparser_evaluate.add_argument('--data-path',
                                    required=True,
                                    type=str,
                                    help='A directory to get data to evaluate.')

    return parser.parse_args()


def init_logging() -> logging.getLoggerClass():
    """
    Init logging. A log level (INFO), format and handler (stdout) are hardcoded.

    :return: a logger
    """
    lg = logging.getLogger('modeling')
    lg.setLevel(logging.INFO)

    fmt = logging.Formatter('%(filename)s:%(lineno)s %(levelname)s %(message)s')

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    lg.addHandler(ch)

    return lg


def create_factors(precision: int) -> dict:
    """
    Create factors for classification, which is simply a mapping between coordinates with the given precision
    (that is 1 degree or 2 degree precision etc.) and a factor (number).
    The bigger precision number, the fewer factors will be.

    :param precision: coordinates precision parameter provided via command line for classification
    or derived from saved classification model.
    :return: dictionary of mapping - scaled coordinates to a factor.
    """
    frs = {}
    n = 0
    for lat in range(int(-90 / precision), int((90 + precision) / precision), 1):
        for lon in range(int(-180 / precision), int((180 + precision) / precision), 1):
            frs[(lat, lon)] = n
            n += 1
    return frs


def make_data(df: pandas.DataFrame,
              file_name: str,
              max_tweet_length: int,
              precision: int = None,
              factors: dict = None) -> List[tf.Tensor]:
    """
    Convert tweet text into numbers, normalize to [0, 1] pad with 0 to fit max tweet length
    in case the text is shorter than that, get coordinates from file name, normalize coordinates to [0,1],
    return list of Tensors.
    The first item in the list is `x` Tensor of shape (N, 1, max_tweet_length),
    where N is the number of dataframe rows processed.
    The second item is `y` Tensor of shape (N, 1, 2) for regression or (N, 1, 1) for classification,
    where N is the number of dataframe rows processed.
    `y` Tensor third dimension is 2 for regression because we want 2 parameters (latitude, longitude) to predict,
    there is dimension 1 for classification because we predict factor.
    Three-dimensional Tensor needed because of how TensorFlow Conv1D layer works.

    :param df: pandas dataframe containing text column to process.
    :param file_name: a file name to get coordinates from its name.
    :param max_tweet_length: command line parameter or derived from saved model.
    :param precision: precision number (1 degree or 2 degree precision etc.) for classification,
    command line parameter or derived from saved model.
    :param factors: classification model factors.
    :return:
    """
    seq = df.loc[:, 'text']
    seq = seq.apply(lambda x: [ord(i) / int('10ffff', 16) for i in list(x)])
    seq = seq.apply(lambda x: x + (max_tweet_length - len(x)) * [0])
    x = tf.constant(list(seq), dtype=tf.float32)
    x = tf.reshape(x, [x.get_shape()[0], 1, x.get_shape()[1]])

    lat, lon = list(map(float, file_name[:-4].split('_')))

    if factors:
        lat = lat // precision
        lon = lon // precision
        y = factors[(lat, lon)]
        y = tf.constant(len(seq) * [y], dtype=tf.float16)
        y = tf.reshape(y, [y.get_shape()[0], 1, 1])
    else:
        y = tf.constant(len(seq) * [[(lat + 90) / 180, (lon + 180) / 360]], dtype=tf.float32)
        y = tf.reshape(y, [y.get_shape()[0], 1, y.get_shape()[1]])

    return [x, y]


def get_data(data_path: str,
             max_tweet_length: int,
             model_type: str,
             test_fraction: int) -> List[list]:
    """
    Read csv files in the given directory and return a length 2 list of Tensors.
    The first item is for training (x, y train Tensors list), the second one is for testing (x, y test Tensors list).
    Tweets longer than max-tweet-length parameter provided via command line will be cut to fit.
    This function simply reads csv files one by one, cut tweets if needed and calls make_data()
    to make the actual work to produce Tensors.

    :param data_path: a directory with csv files.
    :param max_tweet_length: max tweet length, longer than that will be cut.
    :param model_type: classification | regression, command line parameter.
    :param test_fraction: a fraction of data per cent to be used for testing, command line parameter.
    :return: list of Tensors
    """

    factors = None
    if model_type == 'classification':
        factors = create_factors(args.precision)

    # intermediate objects
    x_test_l = []
    y_test_l = []

    x_train_l = []
    y_train_l = []

    # stripping trailing `/` if provided, we add it on our own.
    # the `/` can be or not in the given data path we don't care.
    data_path = data_path.rstrip('/')

    for f in glob.glob(f'{data_path}/*csv'):

        # read as pandas dataframe
        df = pd.read_csv(f, sep=';')

        # ignore empty files
        if df.shape[0] == 0:
            continue

        # cut tweets longer that max tweet length
        df = df[df['text'].map(len) <= max_tweet_length]

        # separate a test part
        df_test = df.sample(frac=test_fraction / 100)

        # take a train fraction
        # in case there were too few rows in the file to get the test fraction (less than 10 rows for instance),
        # simply take the first row for testing.
        if df_test.shape[0] == 0:
            df_test = df.iloc[0, :]
            df_test = df_test.to_frame()
            df_test = df_test.transpose()
            df_train = df.iloc[1:, :]
        else:
            df_train = df.drop(df_test.index)

        # get Tensors
        x, y = make_data(df_test, os.path.basename(f), args.max_tweet_length, args.precision, factors)
        x_test_l.append(x)
        y_test_l.append(y)

        x, y = make_data(df_train, os.path.basename(f), args.max_tweet_length, args.precision, factors)
        x_train_l.append(x)
        y_train_l.append(y)

    x_train = tf.concat(x_train_l, 0)
    y_train = tf.concat(y_train_l, 0)
    x_test = tf.concat(x_test_l, 0)
    y_test = tf.concat(y_test_l, 0)

    return [[x_train, y_train], [x_test, y_test]]


def get_precision(y: int) -> int:
    """
    Derive precision value of saved classification model.
    It takes number of factors (the model's third dimension of output shape) and calculates the precision.
    Based on
    (90/x + 90/x + 1) * (180/x + 180/x + 1) = y
    where x is precision in degrees and y is resulting number of factors given the precision.
    :param y: factors length (classification model third dimension of output shape).
    :return: the model's precision with which it was built.
    """
    x1 = (-540 + math.sqrt(540 * 540 - 4 * (1 - y) * 64800)) / (2 * (1 - y))
    x2 = (-540 - math.sqrt(540 * 540 - 4 * (1 - y) * 64800)) / (2 * (1 - y))
    return int(x1) if x1 > 0 else int(x2)


def mean_error_km(y_true: tf.Tensor, y_pred: tf.Tensor) -> None:
    """
    A dummy function needed as an argument while loading saved regression model.
    Required by TensorFlow.
    :param y_true:
    :param y_pred:
    :return: None
    """
    return None


if __name__ == '__main__':
    args = get_args()
    logger = init_logging()

    if args.subparser_name == 'create':
        logger.info(f'building a {args.layers_number} layer {args.model_type} model, '
                    f'to be saved in {args.model_path}.')
        logger.info(f'max tweet length is limited to {args.max_tweet_length} characters.')
        logger.info(f'gathering data from {args.data_path}.')

        data = get_data(args.data_path,
                        args.max_tweet_length,
                        args.model_type,
                        args.test_fraction)

        try:
            model = models.models_factory(**vars(args))
        except KeyError as err:
            logger.error(f'no such an argument: {err}.')
            raise SystemExit(1)

        history = model.fit(data[0][0],
                            data[0][1],
                            batch_size=32,
                            validation_data=(data[1][0], data[1][1]),
                            epochs=args.epochs,
                            verbose=2)

        if args.charts_directory:
            try:
                charts = model.plot(args.charts_directory, history)
                logger.info(f'saved charts as {charts}')
            except FileNotFoundError as err:
                logger.warning(f'{err}, check that {args.charts_directory} exists and writable.')

        logger.info(f'saving the model as {args.model_path}.')
        model.save(args.model_path,
                   overwrite=True,
                   include_optimizer=True,
                   save_format='tf')

    elif args.subparser_name == 'predict':
        try:
            model = tf.keras.models.load_model(args.model_path, custom_objects={'mean_error_km': mean_error_km})
        except FileNotFoundError as err:
            logger.error(err)
            raise SystemExit(1)

        logger.info(f'loaded {model.name} model, max tweet length is {model.input_shape[2]} for the model.')

        chars = [ord(i) / int('10ffff', 16) for i in list(args.tweet_text)] + \
                 (model.input_shape[2] - len(args.tweet_text)) * [0]
        x = tf.constant(chars, dtype=tf.float32)
        x = tf.reshape(x, [1, 1, tf.shape(x).numpy()[0]])

        if re.search('regression$', model.name):
            m = np.ndarray((1, 1, 2), buffer=np.array([180, 360], dtype=float))
            s = np.ndarray((1, 1, 2), buffer=np.array([90, 180], dtype=float))
            p = model.predict(x) * m - s
            logger.info(f'predicted coordinates: {list(p[0][0])}')

        elif re.search('classification$', model.name):
            precision = get_precision(model.output_shape[2])
            logger.info(f'the model`s precision is {precision} degrees')
            factors_r = {}
            points_l = []
            n = 0
            for lat in range(int(-90 / precision), int((90 + precision) / precision), 1):
                for lon in range(int(-180 / precision), int((180 + precision) / precision), 1):
                    factors_r[n] = [lat * precision, lon * precision]
                    points_l.append([lat * precision, lon * precision])
                    n += 1

            points_l = tf.transpose(tf.constant(points_l))

            logger.info(f'the model`s most confident coordinate: {factors_r[np.argmax(model.predict(x)).real]}.')

            if args.charts_directory:
                pr = model.predict(x)
                plt.clf()
                plt.scatter(points_l[1], points_l[0], s=[1] * len(pr[0][0]), c=1 / pr[0][0], cmap='gray')
                plt.title("Coordinates confidence distribution")
                plt.ylabel('latitude')
                plt.xlabel('longitude')
                plt.grid()
                plt.tight_layout()

                args.charts_directory = args.charts_directory.rstrip('/')
                chart_path = f'{args.charts_directory}/{model.name}_confidence_distribution_{int(time.time())}.png'
                try:
                    plt.savefig(chart_path)
                    logger.info(f'saved confidence distribution as {chart_path}.')
                except FileNotFoundError as err:
                    logger.error(f'{err}, check that {args.charts_directory} exists and writable.')
        else:
            logger.error(f'{model.name} doesn`t match naming convention.')

    elif args.subparser_name == 'evaluate':
        try:
            model = tf.keras.models.load_model(args.model_path,
                                               custom_objects={'mean_error_km': models.mean_error_km})
        except FileNotFoundError as err:
            logger.error(err)
            raise SystemExit(1)

        logger.info(f'loaded {model.name} model, max tweet length is {model.input_shape[2]} for the model.')

        model_type = 'regression'
        factors = None
        precision = None
        if re.search('classification$', model.name):
            model_type = 'classification'
            precision = get_precision(model.output_shape[2])
            logger.info(f'the model`s precision is {precision} degrees')
            factors = create_factors(precision)

        for f in glob.glob(f'{args.data_path}/*csv'):

            df = pd.read_csv(f, sep=';')

            if df.shape[0] == 0:
                continue

            df = df[df['text'].map(len) <= model.input_shape[2]]

            x, y = make_data(df, os.path.basename(f), model.input_shape[2], precision, factors)

        if re.search('regression$', model.name):
            loss, root_mean_squared_error, mean_absolute_error, mean_km_error = model.evaluate(x, y, verbose=0)
            logger.info(f'mean error {round(mean_km_error)} km')
        elif re.search('classification$', model.name):
            loss, accuracy = model.evaluate(x, y, verbose=0)
            logger.info(f'accuracy {accuracy}')
