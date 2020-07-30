import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import random

def decode_text(text):
	return ' '.join([revs_dictionary.get(i,'?') for i in text])

def preprocess(data,for_x=True):
	if for_x:
		final_data = [[j for j in i] for i in data]
	else:
		final_data = [i for i in data]
	return np.array(final_data)

def createCallback():
		os.system('load_ext tensorboard')
		os.makedirs('logs',exist_ok=True)
		logdir = os.path.join('logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
		return keras.callbacks.TensorBoard(logdir)

def display_training(history):
	plt.plot(history.history['val_loss'],label='val_loss')
	plt.plot(history.history['loss'],label='loss')
	plt.legend()
	plt.show()

def display_prediction(model,count,x_test,y_true,full_display=True):
	array = [random.randint(0,len(x_test)) for i in range(count)]
	correct = []
	for i in array:
		prediction = model.predict([x_test[i]])
		prediction = np.argmax(prediction[0])
		actual = np.argmax(y_true[i])
		if full_display:
			print(decode_text(x_test[i]))
			print(f'Prediction : {prediction}')
			print(f'Actual : {actual}')
			print('')

		if prediction == actual:
			correct.append(prediction)
	print(f'Selected {count} random text')
	print(f'Correct Predictions : {len(correct)} out of {count}')
	print(f'Correct rate : {len(correct)/count} ({round(len(correct)/count,2)*100}%)')

def create_model():
	model = keras.Sequential([
			keras.layers.Embedding(10000,16),
			keras.layers.GlobalAveragePooling1D(),
			# keras.layers.Flatten(input_shape=([250,])),
			keras.layers.Dense(32,activation='relu'),
			keras.layers.Dropout(0.2),
			keras.layers.Dense(64,activation='relu'),
			keras.layers.Dropout(0.2),
			keras.layers.Dense(128,activation='relu'),
			keras.layers.Dropout(0.2),
			keras.layers.Dense(2,activation='softmax')
	])
	calback = createCallback()
	earlyStoping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

	model.compile(
		optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy']
	)

	print(model.summary())

	history = model.fit(x_train,y_train,epochs=100,batch_size=32,verbose=1,validation_data=(x_valid,y_valid),callbacks=[calback,earlyStoping])
	display_training(history)

	model.save('NeuralNetworkModel.h5')
	return model

def check_model(model_filename):
	if model_filename in os.listdir():
		model = keras.models.load_model(model_filename)
	else:
		model = create_model()
	return model

def transform_y(y):
	origin = [0,1]
	new = np.array([origin == i for i in y]).astype(np.int)
	return new

data = keras.datasets.imdb
(x_train,y_train),(x_test,y_test) = data.load_data(num_words=10000)

dictionary = data.get_word_index()
dictionary = {k:(v+3) for k,v in dictionary.items()}
dictionary['<PAD>'] = 0
dictionary['<START>'] = 1
dictionary['<UNKNOWN>'] = 2
dictionary['<UNUSED>'] = 3

revs_dictionary = dict([(v,k) for (k,v) in dictionary.items()])

x_train = keras.preprocessing.sequence.pad_sequences(x_train,value=dictionary['<PAD>'],padding='post',maxlen=250)
x_test = keras.preprocessing.sequence.pad_sequences(x_test,value=dictionary['<PAD>'],padding='post',maxlen=250)

x_valid = x_train[:10000]
y_valid = y_train[:10000]
x_train = x_train[10000:]
y_train = y_train[10000:]

y_train = transform_y(y_train)
y_test = transform_y(y_test)
y_valid = transform_y(y_valid)

model = check_model('NeuralNetworkModel.h5')
display_prediction(model,100,x_test,y_test,full_display=False)