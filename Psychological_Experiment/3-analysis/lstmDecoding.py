from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D, Activation, Flatten
import helper as helper
import numpy as np

def decode(raw, event_id, tmin, tmax):
  epochs = helper.getEpochs(raw, event_id, tmin, tmax)
  X = epochs.get_data()

  #swapping features and time points
  X = np.transpose(X, (0, 2, 1)) #prepare X shape to be compatible with LSTM (not sure this is the best way)
  y = epochs.events[:, -1]

  cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

  for train, test in cv.split(X, y):
  	X_train, X_test = X[train], X[test]
  	y_train, y_test = y[train], y[test]

  model = Sequential()

  #add 1-layer cnn (optional)
  model.add(Conv1D(10, kernel_size=3, padding='same', input_shape=(X.shape[1], X.shape[2])))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(MaxPooling1D(padding='same'))

  #perform lstm
  model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2]))) #to feed to next LSTM
  model.add(LSTM(64, return_sequences=True)) #to feed to next LSTM
  model.add(Dropout(0.2))  #prevent overfitting
  model.add(LSTM(32, activation='sigmoid'))
  model.add(Dense(1, activation='softmax'))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   #classification accuracy
  history = model.fit(X_train, y_train, batch_size=16, epochs=20)
  score = model.evaluate(X_test, y_test, batch_size=16)

  print("Accuracy: %.2f%%" % (score[1]*100))
