import tensorflow as tf
from keras.api.layers import GRU, Dropout
from keras.api.models import Sequential 
from utils import *
import sys 
from keras.api.callbacks import ModelCheckpoint, EarlyStopping
from time_history import TimeHistory 

np.set_printoptions(threshold=sys.maxsize)

dataset = np.load('dataset/exp2_final_data.npy')

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

model = Sequential()

#------------- Input layer -------------
model.add(GRU(128, input_shape = (x_train.shape[1:]), activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

#------------- Hidden layers -------------
model.add(GRU(128, activation = 'relu', return_sequences = True))

model.add(GRU(128, activation = 'relu', return_sequences = True))

model.add(GRU(128, activation = 'relu', return_sequences = True))

# ------------- Output layer -------------
model.add(GRU(5, input_shape = (x_train.shape[1:]), activation = 'softmax', return_sequences = True))

opt = tf.keras.optimizers.Adam(learning_rate = 1e-4)

model.compile(loss = 'mse', 
              optimizer = opt, 
              metrics = ['accuracy'])
      
# Checkpoint and early stop
checkpoint = ModelCheckpoint("model_weight_lstm.h5", save_best_only=True, save_weights_only=True, monitor='loss', mode='min', verbose=1)
early_stop = EarlyStopping(monitor='loss', patience=400, mode='min', verbose=1)

# Train the model
time_callback = TimeHistory()
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 2000, shuffle = False, callbacks=[time_callback, checkpoint, early_stop]) 

# Model score
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save('GRU Models/gru_final_epochs()_acc()_loss()_lr()_hl().h5')