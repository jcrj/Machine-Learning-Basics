import cv2 as cv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist #large database of handwritten digits commonly used for training various image processing systems
(x_train, y_train), (x_test, y_test) = mnist.load_data() #loading in the relevant data from mnist datasets

x_train = tf.keras.utils.normalize(x_train, axis = 1) #normalizing a numpy array
x_test = tf.keras.utils.normalize(x_test, axis = 1) #normalizing a numpy array

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
model.add(tf.keras.layers.Dense(units = 256, activation = tf.nn.relu)) #relu == rectified linear unit -> piecewise function that will output in the input directly if it positive, else, output zero
model.add(tf.keras.layers.Dense(units = 256, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax)) #for multiclass predictions (very applicable to this case). Sum of all outputs generated by softmax = 1

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 3) #epochs is the number of times data is ran against the model

accuracy, loss = model.evaluate(x_test, y_test)
print(f' Accuracy: {accuracy}')
print(f' Loss: {loss}')

model.save('digits.model')