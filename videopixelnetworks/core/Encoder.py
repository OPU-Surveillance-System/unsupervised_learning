import keras
from keras import Model
from keras import backend as K
from keras.layers import (Concatenate, ConvLSTM2D, Input, Lambda,
                          TimeDistributed, Conv2D)
from core.layers import ResidualMultiplicativeBlock


class Encoder():

    def __init__(self, filters, frame_shape, frames_count,
                 k=8, lstm_filters=128, rmb_kernel_size=(3,3), lstm_kernel_size=(3,3), lstm_count=1, dilation_rates=[1, 2, 4, 8], return_sequences=True):
        input_sequence = Input(shape=(frames_count, *frame_shape), name='input_sequence')

        lstm_initial_state_input = keras.layers.Input(
            shape=(*frame_shape[:2], 2 * filters),
            name='lstm_initial_state')
        lstm_initial_state = Lambda(lambda x: K.expand_dims(x, axis=1), name='expand_dims')(lstm_initial_state_input)

        X = Lambda(lambda x: x[:,:-1], name='only_first')(input_sequence)
        X = TimeDistributed(Conv2D(
            2 * filters,
            (1, 1),
            strides=(1, 1),
            padding='same',
            activation='tanh'), name='conv2d')(X)

        for i in range(k):
            layer_input = Input(shape=(*frame_shape[:2], 2 * filters))
            layer = Model(
                layer_input,
                ResidualMultiplicativeBlock(
                    filters=filters,
                    dilation_rate=dilation_rates[i % len(dilation_rates)],
                    kernel_size=rmb_kernel_size)(layer_input)
            )
            X = TimeDistributed(layer, name='rmb_%i' % i)(X)
        
        X = Concatenate(axis=1, name='concat')([lstm_initial_state, X])

        for i in range(lstm_count):
            X = ConvLSTM2D(lstm_filters,
                        lstm_kernel_size, strides=(1, 1),
                        activation='tanh',
                        padding='same',
                        return_sequences=return_sequences,
                        name='convlstm2d%i' % (i+1))(X)

        self.model = Model(inputs=[input_sequence, lstm_initial_state_input], outputs=X, name='encoder_network')

    def __call__(self, x):
        return self.model(x)
