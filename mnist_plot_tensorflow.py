#!/bin/env python
import os, sys, dill
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf


def main():
    # 1. set data
    # 2. set model
    N_train = 20000
    N_validation = 4000

    X, Y = get_data()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=N_train)

    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=N_validation)

    n_in = 784
    n_hiddens = [200, 200, 200]
    n_out = 10
    model = DNN(n_in, n_hiddens, n_out)

    history = model.fit(X_train, Y_train, X_validation,
                        Y_validation, epochs=50, batch_size=200, p_keep=0.5)

    print(history['loss'], history['accuracy'])

    # 3. learning the model
    # 4. evaluate the model


def get_data(filename='mnist.pickle', N=None):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            mnist = dill.load(f)
    else:
        mnist = datasets.fetch_mldata('MNIST original', data_home='.')
        with open(filename, 'wb') as g:
            dill.dump(mnist, g)

    n = len(mnist.data)
    indices = np.random.permutation(range(n))[:N]
    X = mnist.data[indices]
    y = mnist.target[indices]
    Y = np.eye(10)[y.astype(int)]

    return X, Y


class DNN(object):

    def __init__(self, n_in, n_hiddens, n_out):
        self.n_in = n_in
        self.n_hiddens = n_hiddens
        self.n_out = n_out
        self.weights = []
        self.biases = []

        self._x = None
        self._t = None
        self._keep_prob = None
        self._sess = None
        self._history = {'accuracy': [], 'loss': [],
                         'val_loss': [], 'val_acc': []}

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    def inference(self, x, keep_prob):
        '''define the model'''
        for i, n_hidden in enumerate(self.n_hiddens):
            if i == 0:
                input = x
                input_dim = self.n_in
            else:
                input = output
                input_dim = self.n_hiddens[i - 1]

            self.weights.append(self.weight_variable([input_dim, n_hidden]))
            self.biases.append(self.bias_variable([n_hidden]))

            h = tf.nn.relu(
                tf.matmul(input, self.weights[-1]) + self.biases[-1])
            output = tf.nn.dropout(h, keep_prob)

        self.weights.append(self.weight_variable(
            [self.n_hiddens[-1], self.n_out]))
        self.biases.append(self.bias_variable([self.n_out]))

        y = tf.nn.softmax(
            tf.matmul(output, self.weights[-1]) + self.biases[-1])

        return y

    def loss(self, y, t):
        '''define the error function'''
        cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(t * tf.log(tf.clip_by_value(y,
                                                       1e-10, 1.0)), reduction_indices=[1])
        )
        return cross_entropy

    def training(self, loss):
        '''define the learning algolithm'''
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train_step = optimizer.minimize(loss)
        return train_step

    def accuracy(self, y, t):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def fit(self, X_train, Y_train,
            X_validation=None, Y_validation=None,
            epochs=100, batch_size=100, p_keep=0.5, verbose=1):
        x = tf.placeholder(tf.float32, shape=[None, self.n_in])
        t = tf.placeholder(tf.float32, shape=[None, self.n_out])
        keep_prob = tf.placeholder(tf.float32)

        self._x = x
        self._t = t
        self.keep_prob = keep_prob

        y = self.inference(x, keep_prob)
        loss = self.loss(y, t)
        train_step = self.training(loss)
        accuracy = self.accuracy(y, t)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        self._sess = sess

        N_train = len(X_train)
        n_batches = N_train // batch_size
        history = {'val_loss': [], 'val_acc': []}
        early_stopping = EarlyStopping(patience=10, verbose=1)
        for epoch in range(epochs):
            X_, Y_ = shuffle(X_train, Y_train)

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                sess.run(train_step, feed_dict={
                         x: X_[start:end], t: Y_[start:end], keep_prob: p_keep})


            loss_ = loss.eval(session=sess, feed_dict={
                              x: X_train, t: Y_train, keep_prob: 1.0})
            accuracy_ = accuracy.eval(session=sess, feed_dict={
                                      x: X_train, t: Y_train, keep_prob: 1.0})

            self._history['loss'].append(loss_)
            self._history['accuracy'].append(accuracy_)

            if verbose:
                print('epoch:', epoch, ' loss:',
                      loss_, ' accuracy:', accuracy_)

            if not X_validation is None:
                val_loss = loss.eval(session=sess, feed_dict={
                                     x: X_validation, t: Y_validation, keep_prob: 1.0})
                val_acc = accuracy.eval(session=sess, feed_dict={
                                        x: X_validation, t: Y_validation, keep_prob: 1.0})

                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                self._history['val_loss'].extend(history['val_loss'])
                self._history['val_acc'].extend(history['val_acc'])


                if early_stopping.validate(val_loss):
                    break


        if not X_validation is None:
            plt.rc('font', family='serif')
            fig = plt.figure()
            ax_acc = fig.add_subplot(111)
            ax_acc.plot(range(epochs), history[
                        'val_acc'], label='acc', color='black')
            ax_loss = ax_acc.twinx()
            ax_loss.plot(range(epochs), history[
                         'val_loss'], label='loss', color='gray')
            plt.xlabel('epochs')
#            plt.ylabel('validation loss')
            plt.legend()
            plt.show()
            # plt.savefig('mnist_tensorflow.eps')
        return self._history

    def evaluate(self, X_test, Y_test):
        return self.accuracy.eval(session=self._sess, feed_dict={self._x: X_test, self._t: Y_test, self._keep_prob: 1.0})


class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False



if __name__ == '__main__':
    main()
