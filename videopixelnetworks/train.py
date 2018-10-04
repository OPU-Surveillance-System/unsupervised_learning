import keras

from core.utils import AucHistory
from core.VideoPixelNetwork import VideoPixelNetwork
from data.DataLoader import DataLoader

frames_count = 10
frames_step = 5  # >= 1
data_loader = DataLoader('data/ped1_train.txt', 'data/ped1_test.txt',
                         frames_count, frames_step, validation_split=0.1)

train_generator = data_loader.train_generator(batch_size=2)
validation_generator = data_loader.validation_generator(batch_size=5)
frame_shape = train_generator.X_shape

filters = 16
k_encoder = 20
lstm_filters = 32
k_decoder = 32
dilation = True
decoder_kernel_size = 5

vpn = VideoPixelNetwork(filters, frame_shape, frames_count,
                        k_encoder=k_encoder,  lstm_filters=lstm_filters,
                        k_decoder=k_decoder, dilation=dilation,
                        decoder_kernel_size=decoder_kernel_size)
model = vpn.model
model.summary()

model.compile(keras.optimizers.Adadelta(),
              loss='categorical_crossentropy')

aucHistory = AucHistory(data_loader, earlyStopping=5)

model.fit_generator(train_generator,
                    validation_data=validation_generator,
                    epochs=10,
                    callbacks=[aucHistory])
