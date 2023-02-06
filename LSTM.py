import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential 
from utils import *
from sklearn.model_selection import train_test_split
import sys 

np.set_printoptions(threshold=sys.maxsize)

dataset = np.load('data.npy')
target = np.zeros(dataset.shape[0])

# test size is a fraction of the data, 20%
# random parameter determines the random number generator used to split the data
x_train, x_test, y_train, y_test = train_test_split(dataset, target, test_size=0.2, random_state=42)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

model = Sequential()

model.add(LSTM(128, input_shape = (x_train.shape[1:]), activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation = 'softmax'))

opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-5)

model.compile(loss = 'sparse_categorical_crossentropy', 
              optimizer = opt, 
              metrics = ['accuracy'])
      
# Train the model
model.fit(x_train, y_train, epochs = 50, validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save('lstm_model.h5')
