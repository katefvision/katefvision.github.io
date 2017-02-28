#!/bin/bash python

import tensorflow as tf
import numpy as np


def create_fc_layer(input, num_neurons, activation, name):
    input_shape = input.get_shape()
    with tf.name_scope(name):
        W = tf.Variable(
            tf.random_normal(
                [input_shape[-1].value, num_neurons], stddev=0.35),
            name='W')
        b = tf.Variable(tf.zeros([num_neurons]), name='b')
        preactivation = tf.matmul(input, W) + b
        output = activation(preactivation)
    return output, preactivation, [W, b]


def create_single_hidden_layer_net(net_name):
    with tf.name_scope(net_name):
        input = tf.placeholder(tf.float32, shape=[None, 784], name='input')
        h_out, h_pre, h_vars = create_fc_layer(input, 100, tf.sigmoid,
                                               'hidden1')
        output, out_pre, out_vars = create_fc_layer(h_out, 10, tf.nn.softmax,
                                                    'output')

    return input, output, h_vars + out_vars


def create_loss(predicted):
    target = tf.placeholder(
        tf.float32, shape=predicted.get_shape(), name='target')
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(
            target * tf.log(predicted), reduction_indices=[1]))

    return target, cross_entropy


def main():
    # construct graph
    input, output, all_vars = create_single_hidden_layer_net('MLP')
    target, cross_entropy = create_loss(output)

    # add summary ops
    tf.summary.scalar('cross_entropy', cross_entropy)
    for var in all_vars:
        tf.summary.histogram(var.name, var)
    all_summaries = tf.summary.merge_all()

    # optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train_op = optimizer.minimize(cross_entropy)

    initializer = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # create a session
    with tf.Session() as sess:
        # initialize all of the variables
        sess.run(initializer)

        # create a summary writer
        # and save the graph definition
        writer = tf.summary.FileWriter('.', sess.graph)
        writer.flush()

        for i in range(10):
            loss, _, summary = sess.run(
                [cross_entropy, train_op, all_summaries],
                feed_dict={
                    input: np.random.randn(10, 784),
                    target: np.random.random_sample([10, 10])
                })
            print(loss)
            writer.add_summary(summary, i)

        writer.flush()
        saver.save(sess, '/tmp/model.ckpt')


if __name__ == '__main__':
    main()
