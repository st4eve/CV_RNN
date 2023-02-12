import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import RNN, LSTMCell, Dense
from tensorflow.keras.backend import dot
from tensorflow.keras import activations, initializers
from keras.models import Sequential

class LSTM(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = 2*units
        self.state_size = [self.units, self.units]

        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('hard_sigmoid')
        self.kernel_initializer = initializers.glorot_uniform(seed=1000)
        self.recurrent_initializer = initializers.get('orthogonal')
        self.bias_initializer = initializers.get('zeros')

        super(LSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.ft = self.add_weight(shape=(input_shape[-1] + self.units, self.units),
                                      initializer=self.kernel_initializer,
                                      name='ft')
        self.ft_b = self.add_weight(shape=(self.units),
                                      initializer=self.bias_initializer,
                                      name='ft_b')
        self.c = self.add_weight(shape=(input_shape[-1] + self.units, self.units),
                                      initializer=self.kernel_initializer,
                                      name='ct')
        self.c_b = self.add_weight(shape=(self.units),
                                      initializer=self.bias_initializer,
                                      name='c_b')
        self.o = self.add_weight(shape=(input_shape[-1] + self.units, self.units),
                                      initializer=self.kernel_initializer,
                                      name='ot')
        self.o_b = self.add_weight(shape=(self.units),
                                      initializer=self.bias_initializer,
                                      name='o_b')
        self.it = self.add_weight(shape=(input_shape[-1] + self.units, self.units),
                                      initializer=self.kernel_initializer,
                                      name='it')
        self.it_b = self.add_weight(shape=(self.units),
                                      initializer=self.bias_initializer,
                                      name='i_b')

        self.built = True

    def call(self, inputs, states):
        h,c = states[0], states[1]

        x = tf.concat([h,inputs], axis=-1)
        y_ft = self.recurrent_activation(dot(x, self.ft) + self.ft_b)
        y_it = self.recurrent_activation(dot(x, self.it) + self.it_b)
        y_c = self.activation(dot(x, self.c) + self.c_b)
        c = c * y_ft + y_it * y_c
        y_o = self.recurrent_activation(dot(x, self.o) + self.o_b)
        h = y_o * self.activation(c)
        output = h
        return output, [h,c]
