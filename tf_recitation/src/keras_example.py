#!/bin/bash python

import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Activation
from keras.models import Model
import keras.backend as K


def create_model(input_size, output_size):
    input = Input(shape=(input_size, ), name='input')
    with tf.name_scope('hidden1'):
        hidden1 = Dense(100, activation='sigmoid')(input)
    with tf.name_scope('output'):
        output = Dense(10, activation='softmax')(input)

    model = Model(input=input, output=output)

    print(model.summary())

    return model


def log_tb_value(name, value):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name

    return summary


def main():
    model = create_model(784, 10)
    model.compile('adam', 'mse', metrics=['mae'])

    sess = K.get_session()
    writer = tf.summary.FileWriter('.', sess.graph)

    for i in range(10):
        mse_loss, mae_metric = model.train_on_batch(
            np.random.randn(10, 784), np.random.random_sample([10, 10]))
        print(mse_loss, mae_metric)

        writer.add_summary(log_tb_value('mse_loss', mse_loss), i)
        writer.add_summary(log_tb_value('mae_metric', mae_metric), i)

    model.save_weights('/tmp/keras_weights')


if __name__ == '__main__':
    main()
