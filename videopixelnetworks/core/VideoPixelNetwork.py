from core.Decoder import Decoder
from core.Encoder import Encoder
from keras import Model
from keras import backend as K
from keras.layers import Input, Lambda


class VideoPixelNetwork():

    def __init__(self, filters, frame_shape, frames_count, k_encoder=8,
                 lstm_filters=128, lstm_count=1, k_decoder=8, dilation=False,
                 decoder_kernel_size=(3, 3)):
        input_sequence = Input(
            shape=(frames_count, *frame_shape),
            name='input_sequence')
        lstm_initial_state = Lambda(
            lambda x: K.zeros((K.shape(x)[0], *frame_shape[:2], 2*filters)),
            name='lstm_initial_state')(input_sequence)

        dilation_rates = [1]
        if dilation:
            dilation_rates = [1, 2, 4, 8]

        self.encoder = Encoder(filters, frame_shape, frames_count, k=k_encoder,
                               lstm_filters=lstm_filters,
                               lstm_count=lstm_count,
                               dilation_rates=dilation_rates)
        self.decoder = Decoder(filters, frame_shape, frames_count, k=k_decoder,
                               lstm_filters=lstm_filters,
                               rmb_kernel_size=decoder_kernel_size)

        H = self.encoder([input_sequence, lstm_initial_state])
        res = self.decoder([input_sequence, H])

        self.model = Model([input_sequence], res)

    def __call__(self, X):
        return self.model(X)

    def summary():
        self.model.summary()
