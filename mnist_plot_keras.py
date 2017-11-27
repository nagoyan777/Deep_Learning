#!/bin/env python
import os, sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.optimizers import SGD

def main():
    n_in = 784
    n_hiddens = [200, 200]
    n_out = 10
    activation = 'relu'
    p_keep = 0.5

    model = Sequential()
    for i, input_dim in enumerate([n_in] + n_hiddens)[:-1]):
        model.add(Dense(n_hiddens[i], input_dim=input_dim))
        model.add(Activation(activation))
        model.add(Dropout(p_keep))

    model.add(Dense(n_out))
    model.add(Activation('softmax'))

if __name__ == '__main__':
    main()
