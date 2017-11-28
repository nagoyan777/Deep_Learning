#!/bin/env python
import os, sys, dill
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.initializers import TruncatedNormal
from keras.layers.normalization import BatchNormalization
from keras import backend as K


def main():

    N_train = 20000
    N_validation = 4000

    X, Y = get_data()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=N_train)

    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=N_validation)

    n_in = 784
    n_hiddens = [200, 200, 200]
    n_out = len(Y[0])
    activation = 'relu'
    p_keep = 0.5

    model = Sequential()
    for i, input_dim in enumerate(([n_in] + n_hiddens)[:-1]):
        model.add(Dense(n_hiddens[i], input_dim=input_dim,
        kernel_initializer=TruncatedNormal(stddev=0.01)))
#         kernel_initializer=weight_variable))
        model.add(BatchNormalization)
        model.add(Activation(activation))
        model.add(Dropout(p_keep))

#    model.add(Dense(n_out), kernel_initializer=weight_variable)
    model.add(Dense(n_out, kernel_initializer=TruncatedNormal(stddev=0.01)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01), metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    epochs = 50
    batch_size = 200

    hist = model.fit(X_train, Y_train, epochs=epochs,
                     batch_size=batch_size,
                     validation_data=(X_validation, Y_validation),
                     callbacks=[early_stopping])

    val_acc = hist.history['val_acc']

    plt.rc('font', family='serif')
    fig = plt.figure()
    plt.plot(range(len(val_acc)), val_acc, label='acc', color='black')
    plt.xlabel('epoch')
    plt.show()
    # plt.savefig('mnist_keras.eps')


def weight_variable(shape):
    return K.truncated_normal(shape, stddev=0.01)


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

if __name__ == '__main__':
    main()
