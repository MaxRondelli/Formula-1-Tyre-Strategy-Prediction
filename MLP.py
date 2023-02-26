import tensorflow as tf
from tensorflow import keras
from keras.layers import GRU, Dropout, Dense
from keras.models import Sequential 
from utils import *
import sys 
from keras.callbacks import ModelCheckpoint, EarlyStopping
from TimeHistory import TimeHistory 
import matplotlib.pyplot as plt 

np.set_printoptions(threshold=sys.maxsize)

dataset = np.load('exp2_final_data.npy')

X = dataset[:, :, :-1]
y = dataset[:, :, -1]

X = X.reshape((78*78, -1))
y = y.flatten()

def train_split(X, y, p):
    N = len(X)
    n_train = int(N*p)
    
    x_train = X[:n_train]
    y_train = y[:n_train]
    x_test = X[n_train:]
    y_test = y[n_train:]

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = train_split(X, y, 0.8)

# --------------------- Model ---------------------
model = Sequential()

model.add(Dense(64, activation = 'relu', input_dim = X.shape[-1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss = 'mse', 
              optimizer = 'adam', 
              metrics = ['accuracy'])
      
# Checkpoint and early stop
checkpoint = ModelCheckpoint("model_weight_lstm.h5", save_best_only=True, save_weights_only=True, monitor='loss', mode='min', verbose=1)
early_stop = EarlyStopping(monitor='loss', patience=400, mode='min', verbose=1)

# Train the model
time_callback = TimeHistory()
hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 100, shuffle = False, callbacks=[time_callback, checkpoint, early_stop]) 

plt.figure()
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training", "Validation"])
plt.savefig('MLP Plots/mlp_plot_epochs()_acc()_loss().png', dpi = 400)


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save('MLP Models/mlp_final_epochs()_acc()_loss().h5')