import time
from keras.api.callbacks import Callback

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.train_time_start = time.time()

    def on_train_end(self, logs={}):
        self.train_time_end = time.time()
        self.total_train_time = self.train_time_end - self.train_time_start
        print('Total training time: {:.2f} seconds'.format(self.total_train_time))
