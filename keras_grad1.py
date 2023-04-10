import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from datetime import datetime

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, units=1):
        super().__init__()
        self.units = units
 
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=True)
 
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer_1 = DenseLayer(128)
        self.layer_2 = DenseLayer(10)
 
    def call(self, inputs):
        x = self.layer_1(inputs)
        x = tf.nn.relu(x)
        x = self.layer_2(x)
        x = tf.nn.softmax(x)
        return x
    
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    
    x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
    x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])
    
    y_train = to_categorical(y_train, 10)
    
    t0 = datetime.now()
    opt = tf.optimizers.Adam(learning_rate=0.001)
    BATCH_SIZE = 32
    EPOCHS = 20
    TOTAL = x_train.shape[0]
    learning_rate=0.01
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
    
    model = NeuralNetwork()
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01),
             loss=tf.losses.categorical_crossentropy,
             metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=20)

    y = model.predict(x_test)
    y2 = tf.argmax(y, axis=1).numpy()
    acc = len(y_test[y_test == y2])/y_test.shape[0] * 100
    print(acc)
    t1 = datetime.now()
    print('время обучения: ', t1-t0)
