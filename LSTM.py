import fastf1 as ff1
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dropout, Dense
from utils import *

# Enable the cache
# ff1.Cache.enable_cache('Cache') # The argument is the name of the folder.

# def load_data():
#     # grand_prix_list = ff1.get_event_schedule(2022)
#     race_list = ['Imola'] # grand_prix_list['Location']
#     x = populate_dataset(race_list)
#     return x 
#     # dataset = generate_dataset(race_list)
#     # np.save('data.npy', dataset)
#     #return dataset

# data = load_data()



# model = keras.Sequential()
# model.add(LSTM(100, input_shape = (10, 28)))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation="sigmoid"))
# model.compile(loss="binary_crossentropy",
#               metrics=[keras.metrics.binary_accuracy],
#               optimizer="adam")
# model.summary()