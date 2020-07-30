import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import datetime

def decode_text(text):
	return ' '.join([dictionary.get(i,'?') for i in text])

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

model = keras.Sequential([
	keras.layers.Embedding(10000,16),
	keras.layers.GlobalAveragePooling1D(),
	keras.layers.Dense(16,activation='relu'),
	keras.layers.Dense(1,activation='sigmoid')
])

calback = createCallback()
earlyStoping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)

model.compile(
	optimizer='adam',
	loss='binary_crossentropy',
	metrics=['accuracy']
)

print(model.summary())

history = model.fit(x_train,y_train,epochs=100,batch_size=32,verbose=1,validation_data=(x_valid,y_valid),callbacks=[calback,earlyStoping])

model.save('NN.h5')

prediction = model.predict(x_test)
print(prediction)