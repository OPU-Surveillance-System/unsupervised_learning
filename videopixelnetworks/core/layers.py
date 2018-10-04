from keras import backend as K
from keras.layers import Activation, Add, Concatenate, Conv2D, Lambda, Multiply
import numpy as np

class MaskedConv2D(Conv2D):

    def __init__(self, filters,
                 kernel_size,
                 mask_type=None,
                 channels_masked=None,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.mask_type = mask_type
        self.channels_masked = channels_masked
        self.stateful = True
        super(MaskedConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
        )

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        if self.mask_type is not None:
            center_h = self.kernel_size[0] // 2
            center_w = self.kernel_size[1] // 2

            mask = np.ones(kernel_shape, dtype='float32')
            mask[center_h, center_w + 1:, :, :] = 0.
            mask[center_h + 1:, :, :, :] = 0.

            if self.mask_type == 'A':
                mask[center_h, center_w, self.channels_masked:, :] = 0.

            self.mask = K.constant(mask, dtype='float32')


        super(MaskedConv2D, self).build(input_shape)
    
    def call(self, inputs):
        if self.mask_type is not None:
            self.kernel = self.kernel * self.mask

            # updates = []
            # res = self.kernel * self.mask
            # updates.append((self.kernel, res))
            # self.add_update(updates, inputs)

        return super(MaskedConv2D, self).call(inputs)

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config['mask_type'] = self.mask_type
        config['channels_masked'] = self.channels_masked
        return config

    def compute_output_shape(self, input_shape):
        return super(MaskedConv2D, self).compute_output_shape(input_shape)

class MultiplicativeUnit():
    n = 0

    def __init__(self, filters=3, kernel_size=(3,3), dilation_rate=1, mask_type=None, channels_masked=0, name=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.mask_type = mask_type
        self.channels_masked = channels_masked

        if name is None:
            self.i = self.n
            self.n += 1
            self.name = 'mu_%i' % self.i
        else:
            self.name = name

        self.g1_conv = MaskedConv2D(
            self.filters,
            self.kernel_size,
            mask_type=self.mask_type,
            channels_masked=self.channels_masked,
            strides=(1, 1),
            dilation_rate=dilation_rate,
            padding='same',
            activation='sigmoid',
            name=self.name + '_g1')
        self.g2_conv = MaskedConv2D(
            self.filters,
            self.kernel_size,
            mask_type=self.mask_type,
            channels_masked=self.channels_masked,
            strides=(1, 1),
            dilation_rate=dilation_rate,
            padding='same',
            activation='sigmoid',
            name=self.name + '_g2')
        self.g3_conv = MaskedConv2D(
            self.filters,
            self.kernel_size,
            mask_type=self.mask_type,
            channels_masked=self.channels_masked,
            strides=(1, 1),
            dilation_rate=dilation_rate,
            padding='same',
            activation='sigmoid',
            name=self.name + '_g3')
        self.u_conv = MaskedConv2D(
            self.filters,
            self.kernel_size,
            mask_type=self.mask_type,
            channels_masked=self.channels_masked,
            strides=(1, 1),
            dilation_rate=dilation_rate,
            padding='same',
            activation='tanh',
            name=self.name + '_u')

    def __call__(self, h):
        g1 = self.g1_conv(h)
        g2 = self.g2_conv(h)
        g3 = self.g3_conv(h)
        u = self.u_conv(h)

        if self.mask_type == 'A':
            g2_h = g2
        else:
            g2_h = Multiply(name=self.name + '_g2_h')([g2, h])

        g3_u = Multiply(name=self.name + '_g3_u')([g3, u])

        output = Activation('tanh', name=self.name + '_tanh')(
            Add(name=self.name + '_add')([
                g2_h,
                g3_u
            ])
        )
        output = Multiply(name=self.name + '_mu')([g1, output])

        return output

class ResidualMultiplicativeBlock():
    n = 0

    def __init__(self, filters=128, channel=1, kernel_size=(3,3), dilation_rate=1, mask=False, first_block=False, last_block=False, name=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.mask = mask
        self.first_block = first_block
        self.last_block = last_block

        if name is None:
            self.i = self.n
            self.n += 1
            self.name = 'rmb_%i' % self.i 
        else:
            self.name = name

        if first_block:
            self.h1_conv = Conv2D(
                self.filters - channel,
                (1, 1),
                strides=(1, 1),
                dilation_rate=dilation_rate,
                padding='same',
                activation='tanh',
                name=self.name + '_h1')
        else:
            self.h1_conv = Conv2D(
                self.filters,
                (1, 1),
                strides=(1, 1),
                dilation_rate=dilation_rate,
                padding='same',
                activation='tanh',
                name=self.name + '_h1')

        if mask:
            if first_block:
                self.h2_mu = MultiplicativeUnit(
                    filters=self.filters,
                    kernel_size=(3,3),
                    dilation_rate=dilation_rate,
                    mask_type='A',
                    name=self.name + '_h2_mu')
            else:
                self.h2_mu = MultiplicativeUnit(
                    filters=self.filters,
                    kernel_size=(3,3),
                    dilation_rate=dilation_rate,
                    mask_type='B',
                    name=self.name + '_h2_mu')
            
            self.h3_mu = MultiplicativeUnit(
                filters=self.filters,
                kernel_size=(3,3),
                dilation_rate=dilation_rate,
                mask_type='B',
                name=self.name + '_h3_mu')


        else:
            self.h2_mu = MultiplicativeUnit(
                filters=filters,
                kernel_size=(3,3),
                dilation_rate=dilation_rate,
                mask_type=None,
                name=self.name + '_h2_mu')
            self.h3_mu = MultiplicativeUnit(
                filters=filters,
                kernel_size=(3,3),
                dilation_rate=dilation_rate,
                mask_type=None,
                name=self.name + '_h3_mu')

        if last_block:
            self.h4_conv = Conv2D(
                256 * channel,
                (1, 1),
                strides=(1, 1),
                dilation_rate=dilation_rate,
                padding='same',
                activation='softmax',
                name=self.name + '_h4')
        else:
            self.h4_conv = Conv2D(
                2 * self.filters,
                (1, 1),
                strides=(1, 1),
                dilation_rate=dilation_rate,
                padding='same',
                activation='tanh',
                name=self.name + '_h4')

    def __call__(self, h, x=None):
        if self.first_block and x is None:
            raise Exception('if first_block is true, x must be passed when rmb is called')

        h1 = self.h1_conv(h)

        if self.first_block:
            h1 = Concatenate(axis=3)([x, h1])

        h2 = self.h2_mu(h1)
        h3 = self.h3_mu(h2)
        h4 = self.h4_conv(h3)

        if self.last_block:
            output = h4
        else:
            output = Add(name=self.name + '_add')([h, h4])

        return output