import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
import tensorflow as tf

class DenseNN(tf.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.fl_init = False
 
    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name="w")
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name="b")
 
            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)
 
            self.fl_init = True
 
        y = x @ self.w + self.b
        return y
    def fit(self, X_train, y_train, epochs=50):
        loss = lambda x, y: tf.reduce_mean(tf.square(x-y))
        opt = tf.optimizers.Adam(learning_rate=0.01)
        EPOCHS = epochs
        for n in range(EPOCHS):
            for x, y in zip(X_train, y_train):
                x = tf.expand_dims(x, axis=0)
                y = tf.constant(y, shape=(1, 1))
        
                with tf.GradientTape() as tape:
                    f_loss = loss(y, model(x))
        
                grads = tape.gradient(f_loss, model.trainable_variables)
                opt.apply_gradients(zip(grads, model.trainable_variables))
        
            print(f_loss.numpy())


    
if __name__ == '__main__':
    model = DenseNN(1)
    print(model(tf.constant([[1.0, 2.0]])))
    X_train = tf.random.uniform(minval=0, maxval=10, shape=(100,2))
    y_train = [2*a+b for a, b in X_train]
    print(type(y_train[0]))
    model.fit(X_train, y_train)

    print(model.trainable_variables)