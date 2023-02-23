import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from TimeHistory import TimeHistory 
from utils import *
import sys 
from keras.callbacks import ModelCheckpoint, EarlyStopping

# np.set_printoptions(threshold=sys.maxsize)

dataset = np.load('exp2_final_data.npy')
#print(dataset.shape)

X = dataset[:, :, :-1]
y = dataset[:, :, -1]

def train_split(X, y, p):
    N = len(X)
    n_train = int(N*p)
    
    x_train = X[:n_train]
    y_train = y[:n_train]
    x_test = X[n_train:]
    y_test = y[n_train:]

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = train_split(X, y, 0.8)
y_train = tf.keras.utils.to_categorical(y_train, 5)
y_test = tf.keras.utils.to_categorical(y_test, 5)

'''
p_i = calcolare la probabilità di ogni compound all'interno di y_train 

p_0 = np.sum((y_train == 0)) / len(y_train)
n_0 = np.sum((y_test == 0))

'''
for i in y_train:
    p_i = np.sum((y_train == i)) / len(y_train)
    n_i = np.sum((y_train == i))
    
random_classificator = (p_i * n_i) / n_i

model = Sequential()

#------------- Input layer -------------
model.add(LSTM(128, input_shape = (x_train.shape[1:]), activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

#------------- Hidden layers -------------
model.add(LSTM(128, activation = 'relu', return_sequences = True))
# model.add(Dropout(0.2))
model.add(LSTM(128, activation = 'relu', return_sequences = True))

# model.add(LSTM(128, activation = 'relu', return_sequences = True))
# model.add(Dropout(0.2))

# ------------- Output layer -------------
model.add(LSTM(5, input_shape = (x_train.shape[1:]), activation = 'softmax', return_sequences = True))

opt = tf.keras.optimizers.Adam(learning_rate = 5e-7)

model.compile(loss = 'mse', 
              optimizer = opt, 
              metrics = ['accuracy'])
      
# Checkpoint and early stop
checkpoint = ModelCheckpoint("model_weight_lstm.h5", save_best_only=True, save_weights_only=True, monitor='loss', mode='min', verbose=1)
early_stop = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1)

# Train the model
time_callback = TimeHistory()
model.fit(x_train, y_train, epochs = 1, shuffle = False, callbacks=[time_callback]) #, callbacks=[checkpoint, early_stop])

score = model.evaluate(x_test, y_test, verbose=0)
print('Random classificator:', random_classificator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save('LSTM Models/lstm_final_epochs()_acc()_loss()_lr().h5')