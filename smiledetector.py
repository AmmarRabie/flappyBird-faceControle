import numpy as np
import scipy.misc as mc


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.models import load_model



'''
from smiledetector import SmileDetector
detector = SmileDetector()
detector.predict(img)
'''


class SmileDetector:

	def __init__(self, path='weights.h5', mean='mean.npy'):
		self.model = Sequential()
		self.build_architecture()
		self.compile()
		self.load_weights(path)
		self.mean = np.load(mean)


	def build_architecture(self):
		self.model.add(Conv2D(32, kernel_size=(7, 7), padding="same", activation="relu", kernel_regularizer=l2(0.0001)))
		self.model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu", strides=(2, 2)))
		self.model.add(Conv2D(64, kernel_size=(5, 5), padding="same", activation="relu", kernel_regularizer=l2(0.0001)))
		self.model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", strides=(2, 2)))
		self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer=l2(0.0001)))
		self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer=l2(0.0001)))
		self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", kernel_regularizer=l2(0.0001)))
		self.model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu", strides=(2, 2)))
		self.model.add(Flatten())
		self.model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.001)))
		self.model.add(Dropout(0.5))
		self.model.add(Dense(2, activation="softmax"))

	def compile(self):
		self.adam = Adam(lr = 0.00001)

		self.model.compile(optimizer=self.adam,
		              loss='categorical_crossentropy',
		              metrics=['accuracy'])
		x = np.ones((1, 64, 64, 3))
		y = np.ones((1,))

		self.model.fit(x, to_categorical(y), epochs=1, verbose=0)

	def load_weights(self, path):
		self.model.load_weights(path)

	def preprocess_image(self, img):
		img = mc.imresize(img, (64, 64, 3), interp='bicubic')
		img = img + self.mean
		img_array = np.empty((1, 64, 64, 3))
		img_array[0] = img
		return img_array

	def predict(self, img):
		img = self.preprocess_image(img)
		return np.argmax(self.model.predict(img)[0])
