import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dropout, Dense
from keras.models import Sequential
from TimeHistory import TimeHistory 
from utils import *
import sys 
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt 

np.set_printoptions(threshold=sys.maxsize)

dataset = np.load('exp2_final_data.npy')
print(dataset.shape)

X = dataset[:, :, :-1]
y = dataset[:, :, -1]

# X = X.reshape((78*78, -1))
# y = y.flatten()

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

# p_i = np.mean(y_train, axis = 0)
# n_i = np.sum(y_test, axis = 0)

# blind_classifier = np.sum(p_i * n_i) / np.sum(n_i)

'''
senza nn, l'accaratezza sparando a caso prendi il 25%. sparando a caso significa che con il 14% prevedi soft, 29% prevedi medium etc. 
sparando a caso prevedi le % nel training set.
'''

model = Sequential()

#------------- Input layer -------------
model.add(LSTM(128, input_shape = (x_train.shape[1:]), activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

#------------- Hidden layers -------------
model.add(LSTM(128, activation = 'relu', return_sequences = True))
# model.add(Dropout(0.2))
model.add(LSTM(128, activation = 'relu', return_sequences = True))

model.add(LSTM(128, activation = 'relu', return_sequences = True))
# model.add(Dropout(0.2))

# ------------- Output layer -------------
model.add(LSTM(5, input_shape = (x_train.shape[1:]), activation = 'softmax', return_sequences = True))

opt = tf.keras.optimizers.Adam(learning_rate = 1e-4)
# 1e-4 , 5e-4
model.compile(loss = 'mse', 
              optimizer = opt, 
              metrics = ['accuracy'])
      
# Checkpoint and early stop
checkpoint = ModelCheckpoint("model_weight_lstm.h5", save_best_only=True, save_weights_only=True, monitor='loss', mode='min', verbose=1)
early_stop = EarlyStopping(monitor='loss', patience=400, mode='min', verbose=1)

# Train the model
time_callback = TimeHistory()
hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 2000, shuffle = False, callbacks=[time_callback, checkpoint, early_stop]) 

plt.figure()
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"])
plt.savefig('LSTM Plots/lstm_plot_epochs()_acc()_loss()_lr().png', dpi = 400)

score = model.evaluate(x_test, y_test, verbose=0)

# print("p_i:", p_i)
# print("n_i:", n_i)
# print('Blind classifier:', blind_classifier)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save('LSTM Models/lstm_final_epochs()_acc()_loss()_lr().hl().h5')