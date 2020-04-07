from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
import helper as helper
import numpy as np
from tcn import TCN

def decode(raw, event_id, tmin, tmax):
	epochs = helper.getEpochs(raw, event_id, tmin, tmax)
	X = epochs.get_data()

	#swapping features and time points
	X = np.transpose(X, (0, 2, 1)) #prepare X shape to be compatible with LSTM (not sure this is the best way)
	y = epochs.events[:, -1]

	cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2)

	for train, test in cv.split(X, y):
		X_train, X_test = X[train], X[test]
		y_train, y_test = y[train], y[test]

	print(X.shape[1])
	print(X.shape[2])
	batch_size, timesteps, input_dim = None, X.shape[1], 4

	i = Input(batch_shape=(batch_size, timesteps, input_dim))
	o = TCN(return_sequences=False)(i)
	o = Dense(1)(o)
	m = Model(inputs=[i], outputs=[o])
	m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	history = m.fit(X_train, y_train, validation_split=0.2, epochs=5)
	score = m.evaluate(X_test, y_test, batch_size=16)

	print("Accuracy: %.2f%%" % (score[1]*100))