import tensorflow as tf
from tensorflow.keras import layers
model=tf.keras.Sequential([
    layers.Dense(units=1,input_shape=[1])
])
model.compile(optimizer='sgd',loss="mean_squared_error")
print("神经网络已创建！结构如下：")
model.summary()