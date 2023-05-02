import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from datetime import datetime
from my_optim import SGOptimizer
from my_MomentumOpt_tree import MyMomentumOptimizer_tree
from gp_tree_tensor_TF import *
from gp_tree_tensor_TF2 import *
from gp_tree_tensor_АdaGrad import tree_AdaGrad
from gp_tree_tensor_RMSProp import tree_RMSProp

    
if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    
    x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
    x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])
    
    y_train = to_categorical(y_train, 10)
    
    t0 = datetime.now()
    opt = tf.optimizers.Adam(learning_rate=0.01)
    opt1 = SGOptimizer()
    opt2 = MyMomentumOptimizer_tree(learning_rate=0.01, tree=tree12)
    opt3 = MyMomentumOptimizer_tree(learning_rate=0.01, tree=tree_grad)
    opt4 = MyMomentumOptimizer_tree(learning_rate=0.01, tree=tree_AdaGrad)
    opt_RMSProp = MyMomentumOptimizer_tree(learning_rate=0.001, tree=tree_RMSProp)
    opt5=tf.keras.optimizers.Adagrad(learning_rate=0.01)
    opt_RMSProp_TF=tf.keras.optimizers.RMSprop(learning_rate=0.01)

    BATCH_SIZE = 32
    EPOCHS = 10
    TOTAL = x_train.shape[0]
    learning_rate=0.01
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
    model = Sequential()
    model.add(Dense(128, input_dim=28*28,  activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=opt_RMSProp_TF,
                loss=tf.losses.categorical_crossentropy,
                metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS)

    y = model.predict(x_test)
    y2 = tf.argmax(y, axis=1).numpy()
    acc = len(y_test[y_test == y2])/y_test.shape[0] * 100
    print(acc)
    t1 = datetime.now()
    print('время обучения: ', t1-t0)
