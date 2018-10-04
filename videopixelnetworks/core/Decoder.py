import numpy as np

import keras
from keras import Model
from keras import backend as K
from keras.layers import Concatenate, Input, Lambda, Reshape, TimeDistributed, Conv2D

from core.layers import ResidualMultiplicativeBlock

class Decoder():

    def __init__(self, filters, frame_shape, frames_count, k=8, lstm_filters=128, rmb_kernel_size=(3, 3)):
        self.filters = filters
        self.frame_shape = frame_shape
        self.frames_count = frames_count
        self.rmb_kernel_size = rmb_kernel_size

        input_sequence = Input(shape=(self.frames_count, *self.frame_shape), name='input_sequence')
        context = Input(shape=(self.frames_count, *self.frame_shape[:2], lstm_filters), name='context')

        context_rmb = self.rmb_without_mask(context, self.filters, 'init_rmb')

        hx = Concatenate(axis=4)([context_rmb, input_sequence])

        hx_input = Input(shape=(*self.frame_shape[:2], self.frame_shape[2] + self.filters*2))
        h, x = Lambda(lambda hx: [hx[:, :, :, :self.filters*2], hx[:, :, :, self.filters*2:]])(hx_input)
        layer = Model(
            hx_input,
            ResidualMultiplicativeBlock(
                filters=self.filters,
                mask=True,
                channel=self.frame_shape[2],
                first_block=True,
                kernel_size=self.rmb_kernel_size)(h, x)
        )
        X = TimeDistributed(layer, name='rmb_0')(hx)

        for i in range(2, k-1):
            X = self.rmb_with_mask(X, self.filters, (*self.frame_shape[:2], self.filters * 2), 'rmb_%i' % i)
        
        X = self.rmb_with_mask(X, self.filters, (*self.frame_shape[:2], self.filters * 2), 'rmb_%i' % (k-1), last_block=True, channel=frame_shape[-1])

        if self.frame_shape[-1] > 1:
            X = Reshape((self.frames_count, *self.frame_shape, 256), name='final_reshape')(X)
        
        res = X
        self.model = Model([input_sequence, context], res, name='decoder_network')

    def rmb_without_mask(self, context, filters, name):
        layer_input = Input(shape=(*self.frame_shape[:2], 2*filters))

        context = TimeDistributed(Conv2D(
            2 * filters,
            (1, 1),
            strides=(1, 1),
            padding='same',
            activation='tanh'), name='conv2d')(context)

        layer = Model(
            layer_input,
            ResidualMultiplicativeBlock(
                filters=filters,
                kernel_size=self.rmb_kernel_size)(layer_input)
        )
        return TimeDistributed(layer, name=name)(context)

    def rmb_with_mask(self, X, filters, shape, name, last_block=False, channel=1):
        layer_input = Input(shape=shape)
        layer = Model(
            layer_input,
            ResidualMultiplicativeBlock(
                filters=filters,
                mask=True,
                kernel_size=self.rmb_kernel_size,
                last_block=last_block,
                channel=channel)(layer_input)
        )
        return TimeDistributed(layer, name=name)(X)

    def __call__(self, x):
        return self.model(x)