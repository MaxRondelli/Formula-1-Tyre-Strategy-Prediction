import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential 
from Utils import *
import sys 
from keras.callbacks import ModelCheckpoint, EarlyStopping
from TimeHistory import TimeHistory 

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
y_train = tf.keras.utils.to_categorical(y_train, 5)
y_test = tf.keras.utils.to_categorical(y_test, 5)

# --------------------- Model ---------------------
model = Sequential()

model.add(Dense(128, activation = 'relu', input_dim = X.shape[-1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation = 'relu', input_dim = X.shape[-1]))

# -------- Output layer --------
model.add(Dense(5, activation='softmax'))

model.compile(loss = 'mse', 
              optimizer = 'adam', 
              metrics = ['accuracy'])
      
# Checkpoint and early stop
checkpoint = ModelCheckpoint("model_weight_lstm.h5", save_best_only=True, save_weights_only=True, monitor='loss', mode='min', verbose=1)
early_stop = EarlyStopping(monitor='loss', patience=400, mode='min', verbose=1)

print(y_train.shape)

# Train the model
time_callback = TimeHistory()
hist = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 2000, shuffle = False) #, callbacks=[time_callback, checkpoint, early_stop]) 

# Model score
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save('MLP Models/mlp_final_epochs()_acc()_loss().h5')