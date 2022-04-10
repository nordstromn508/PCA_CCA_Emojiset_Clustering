"""
models.py
    A collection of different machine learning algorithm implementations for importing

    @author: Nicholas Nordstrom
"""
import time
from tensorflow.keras import models, layers
from tensorflow.python.keras.applications import densenet, mobilenet, inception_v3, efficientnet


def mobile_net(input_shape, output_shape, verbose=False, loss='binary_crossentropy', activation='softmax',
               optimizer='adam', metrics=['accuracy']):
    """
    :param loss: loss function to calculate loss between epochs
    :@author: https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    creates our algorithm to learn from our dataset.
    :param input_shape: shape of input for model
    :param output_shape: shape of output
    :param verbose: option to print details about model
    :return: the model object.
    """
    start = time.time()
    model = mobilenet.MobileNet(
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=output_shape,
        classifier_activation=activation)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if verbose:
        model.summary()
    return model, time.time() - start


def inception_v3(input_shape, output_shape, verbose=False, loss='binary_crossentropy', activation='softmax', optimizer='adam', metrics=['accuracy']):
    """
    :param loss: loss function to calculate loss between epochs
    :@author: https://docs.w3cub.com/tensorflow~python/tf/keras/applications/inceptionV3
    Creates an InceptionV3 model
    :param input_shape: shape of input layer
    :param output_shape: shape of output layer
    :param verbose: option to print model summary to console
    :return: compiled and ready-to-train model
    """

    start = time.time()
    model = inception_v3.InceptionV3(
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=output_shape,
        classifier_activation=activation)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if verbose:
        model.summary()
    return model, time.time() - start


def dense_net201(input_shape, output_shape, verbose=False, loss='binary_crossentropy', activation='softmax', optimizer='adam', metrics=['accuracy']):
    """
    :param loss: loss function to calculate loss between epochs
    :@author: https://docs.w3cub.com/tensorflow~python/tf/keras/applications/densenet201
    Creates a DenseNet201 model
    :param input_shape: shape of input layer
    :param output_shape: shape of output layer
    :param verbose: option to print model summary to console
    :return: compiled and ready-to-train model
    """
    start = time.time()
    model = models.sequential.Sequential()
    model.add(densenet.DenseNet201(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=output_shape))

    model.add(layers.Flatten())
    model.add(layers.Dense(output_shape, activation=activation))

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if verbose:
        model.summary()
    return model, time.time() - start


def efficient_net(input_shape, output_shape, verbose=False, loss='binary_crossentropy', activation='softmax', optimizer='adam', metrics=['accuracy']):
    """
    :param loss: loss function to calculate loss between functions
    :@author: https://towardsdatascience.com/an-in-depth-efficientnet-tutorial-using-tensorflow-how-to-use-efficientnet-on-a-custom-dataset-1cab0997f65c
    Creates a efficientNet model, loads trained weights as a starting point
    :param input_shape: shape of input
    :param output_shape: shape of output
    :param verbose: option to print model summary
    :return: compiled model
    """
    start = time.time()

    model = models.sequential.Sequential()
    model.add(efficientnet.EfficientNetB6(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=output_shape))

    model.add(layers.Flatten())
    model.add(layers.Dense(output_shape, activation=activation))

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)
    if verbose:
        model.summary()
    return model, time.time() - start


def vgg16(input_shape, output_shape, verbose=False, loss='binary_crossentropy', activation='softmax', optimizer='adam', metrics=['accuracy']):
    """
    :param loss: loss function to calculate loss between epochs
    :@author: https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
    creates our algorithm to learn from our dataset.
    :param input_shape: shape of input for model
    :param output_shape: shape of output
    :param verbose: option to print details about model
    :return: the model object.
    """
    start = time.time()
    model = vgg16.VGG16(
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=output_shape,
        classifier_activation=activation)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if verbose:
        model.summary()
    return model, time.time() - start


def kmeans():
    """
    TODO: Implement
    :return:
    """
    pass

