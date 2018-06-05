# This neural network is the the Q-function, run it like this:
# model.predict(state.reshape(1,64), batch_size=1)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adam , SGD


def LSTM_model(input_dim):
    batch_size = 1
    num_features = input_dim

    model = Sequential()
    model.add(LSTM(64,
                   input_shape=(1, num_features),
                   return_sequences=True,
                   stateful=False))
    model.add(Dropout(0.5))

    model.add(LSTM(64,
                   input_shape=(1, num_features),
                   return_sequences=False,
                   stateful=False))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

    rms = RMSprop()
    adam = Adam()
    model.compile(loss='mse', optimizer=adam)
    return model

def MLP_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model
