import matplotlib.image as mpimg
import numpy as np
from keras.utils import Sequence, to_categorical


class DataLoader():
    
    def __init__(self, train_file, test_file, frames_count, frames_step, validation_split=0.0):
        self.train_file = train_file
        self.test_file = test_file
        self.validation_split = validation_split
        self.frames_count = frames_count
        self.frames_step = frames_step

        if self.validation_split < 0 or self.validation_split > 1:
            raise "Invalid validation_split"

        self.read_files()
        self.split()

    def read_files(self):
        # Training data
        with open(self.train_file, 'r') as f:
            training_path = [f for f in map(lambda s: s.strip(),f.readlines())]

        self.train_videos = {}
        for path in training_path:
            split = path.split('/')
            video = '/'.join(split[:-1])
            frame = split[-1]

            if self.train_videos.get(video) is None:
                self.train_videos[video] = []

            self.train_videos[video].append(frame)

        # Test data
        with open(self.test_file, 'r') as f:
            test_path = [f for f in map(lambda s: tuple(s.strip().split()),f.readlines())]
        
        self.test_videos = {}
        self.normal_frames = {}
        self.abnormal_frames = {}

        for (path, abnormal) in test_path:
            split = path.split('/')
            video = '/'.join(split[:-1])
            frame = split[-1]

            if self.test_videos.get(video) is None:
                self.test_videos[video] = []

            self.test_videos[video].append(frame)
            if int(abnormal):
                if self.abnormal_frames.get(video) is None:
                    self.abnormal_frames[video] = []
                self.abnormal_frames[video].append(frame)
            else:
                if self.normal_frames.get(video) is None:
                    self.normal_frames[video] = []
                self.normal_frames[video].append(frame)


    def split(self):
        if self.validation_split > 0:
            n = len(self.train_videos)
            num_val = int(n * self.validation_split)
            num_train = n - num_val

            keys = np.array(list(self.train_videos.keys()))
            val_idx = np.random.choice(range(0, n), num_val, replace=False)

            val_keys = keys[val_idx]
            train_keys = np.delete(keys, val_idx, axis=0)

            videos = self.train_videos
            self.train_videos = {}
            self.validation_videos = {}

            for k in train_keys:
                self.train_videos[k] = videos[k]
            for k in val_keys:
                self.validation_videos[k] = videos[k]

    def train_generator(self, **kwargs):
        prefix = '/'.join(self.train_file.split('/')[:-1]) + '/'
        return DataGenerator(self.train_videos, sequenceLength=self.frames_count, sequenceStep=self.frames_step, pathPrefix=prefix, **kwargs)

    def validation_generator(self, **kwargs):
        if self.validation_split == 0.0:
            raise "Validation split set to zero !"
        prefix = '/'.join(self.train_file.split('/')[:-1]) + '/'
        return DataGenerator(self.validation_videos, sequenceLength=self.frames_count, sequenceStep=self.frames_step, pathPrefix=prefix, **kwargs)

    def test_generator(self, only_normal=None, **kwargs):
        # if only_normal None : every test sequences
        # if only_normal true : only sequences with normal at the end
        # if only_normal false : only sequences with abnormal at the end

        prefix = '/'.join(self.test_file.split('/')[:-1]) + '/'
        f = None
        if only_normal is not None:
            f = self.normal_frames if only_normal else self.abnormal_frames
        return DataGenerator(self.test_videos, sequenceLength=self.frames_count, sequenceStep=self.frames_step, pathPrefix=prefix, data_filter=f, **kwargs)

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, videos, sequenceLength=10, sequenceStep=1, batch_size=32, shuffle=True, pathPrefix='', data_filter=None, filter_ratio=1):
        'Initialization'
        self.videos = videos
        self.sequenceLength = sequenceLength
        self.sequenceStep = sequenceStep
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pathPrefix = pathPrefix

        self.data = []
        self.totalLength = 0

        self.totalSequenceLength = self.sequenceStep * (self.sequenceLength - 1) + 1

        for key, frames in self.videos.items():
            self.totalLength += len(frames)

            choice = []
            if data_filter is None:
                choice = list(map(lambda pos: (key, pos), range(len(frames[:-self.totalSequenceLength+1]))))
            elif data_filter.get(key) is not None:
                for pos in range(len(frames[:-self.totalSequenceLength+1])):
                    count = 0
                    for j in range(pos, pos+self.totalSequenceLength):
                        if frames[j] in data_filter.get(key):
                            count += 1

                    if count >= self.totalSequenceLength * filter_ratio:
                        choice.append((key, pos))

            self.data += choice

        self.len = int(np.ceil(len(self.data) / self.batch_size))
        
        k = list(self.videos.keys())[0]
        sample_filepath = self.videos[k][0]
        sample = self.read_frame(k, sample_filepath)
        self.data_shape = sample.shape

        if len(self.data_shape) == 2:
            self.X_shape = (*self.data_shape, 1)
            self.Y_shape = (*self.data_shape, 256)
        else:
            self.X_shape = self.data_shape
            self.Y_shape = (*self.data_shape, 256)

        self.on_epoch_end()

    def read_frame(self, k, f):
        return mpimg.imread(self.pathPrefix + k + '/' + f)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.len

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        data_temp = [self.data[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(data_temp)

        return X, Y

    def __data_generation(self, data_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((len(data_temp), self.sequenceLength, *self.X_shape))
        Y = np.empty((len(data_temp), self.sequenceLength, *self.Y_shape))
        
        # Generate data
        for i, datum in enumerate(data_temp):
            key, pos = datum
            frames = self.videos[key][pos:pos+self.totalSequenceLength]
            k = 0
            for j, frame in enumerate(frames):
                if j % self.sequenceStep == 0:
                    raw = self.read_frame(key, frame)
                    X[i, k,] = raw.reshape(self.X_shape)
                    Y[i, k,] = to_categorical(raw, num_classes=256)
                    k += 1


        return X, Y
