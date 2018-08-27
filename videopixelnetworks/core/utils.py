import time
import numpy as np
import sklearn.metrics
from keras.callbacks import Callback

def compute_auc(model, data_loader):
    normal_generator = data_loader.test_generator(batch_size=20, only_normal=True)
    abnormal_generator = data_loader.test_generator(batch_size=20, only_normal=False)

    y_test = [] # True, False
    y_prob = [] # Score
    y_mse = []

    epsilon = 0.001

    c = 1
    total_i = len(normal_generator) + len(abnormal_generator)

    start_time = time.time()
    for generator, case in [(normal_generator, False), (abnormal_generator, True)]:
        for i in range(len(generator)):
            elapsed = time.time() - start_time
            eta = (total_i * elapsed) / c - elapsed
            print('(%i) %i/%i, elapsed : %is, eta: %im%is' % (case, c, total_i, elapsed, eta // 60, eta % 60), end=' '*10 + '\r')
            X, Y = generator.__getitem__(i)
            
            if (X.shape[0] == 0):
                break
            
            P = model.predict(X)
            P = np.array(P)
            Y = np.array(Y)

            P_argmax = np.argmax(P, axis=4)
            Y_argmax = np.argmax(Y, axis=4)

            scores_log = -np.log(np.max(Y * P, axis=4) + epsilon).sum(axis=(1, 2, 3))
            scores_mse = np.sum(np.square(P_argmax - Y_argmax), axis=(1,2,3))

            y_test += [case] * len(scores_log)
            y_prob += list(scores_log)
            y_mse += list(scores_mse)

            c += 1

    out = []

    for y_score, name in [(y_prob, 'log_likelihood'), (y_mse, 'mse')]:
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_score)
        auc = sklearn.metrics.auc(fpr, tpr)
        out.append((auc, fpr, tpr, thresholds, name))
        print('(%s) auc=%0.4f computed in %0.0fs' % (name, auc, elapsed), ' '*20)

    return out

class AucHistory(Callback):

    def __init__(self, data_loader, earlyStopping=1, before=False, paramsString='', weightsFile='weights/'):
        self.data_loader = data_loader
        self.earlyStopping = earlyStopping
        self.before = before
        self.paramsString = paramsString
        self.weightsFile = weightsFile
        if not weightsFile.endswith('/'):
            self.weightsFile = '%s/' % weightsFile

        self.lastEpoch = 0

    def on_train_begin(self, logs={}):
        self.history = []
        self.best = 0
        self.best_epoch = 0
        self.wait = 0

        if self.before:
            self.on_epoch_end(-1)

    def on_epoch_end(self, epoch, logs={}):
        out = compute_auc(self.model, self.data_loader)
        self.history.append(out)

        auc, fpr, tpr, thresholds, name = (0 , None, None, None, '')
        for i in range(len(out)):
            if out[i][0] > auc:
                auc, fpr, tpr, thresholds, name = out[i]

        if auc > self.best:
            self.best = auc
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1

        filepath = '%svpn-%s-%i-%f-%s.h5' % (self.weightsFile, self.paramsString, epoch + 1, auc, name)
        self.model.save_weights(filepath)

        self.lastEpoch = epoch + 1

        if self.earlyStopping > 0 and self.wait >= self.earlyStopping:
            self.model.stop_training = True
            print('EarlyStopping ! best_auc: %0.5f, epoch: %i' % (self.best, self.best_epoch + 1))
