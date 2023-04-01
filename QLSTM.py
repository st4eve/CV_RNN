from CV_quantum_layers import QuantumLayer_MultiQunode, Activation_Layer
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import RNN, LSTMCell, Dense
from tensorflow.keras.backend import dot
from tensorflow.keras import activations, initializers
from keras.models import Sequential
tf.random.set_seed(100)

class QLSTM(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = [self.units, self.units]
        
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('hard_sigmoid')
        self.kernel_initializer = initializers.get('glorot_uniform')
        self.f_q = QuantumLayer_MultiQunode(
                                            self.units,
                                            1,
                                            1,
                                            5,
                                            encoding_method="Amplitude_Phase",
                                            regularizer=regularizers.L1(l1=0.01),
                                            max_initial_weight=0.28499999999999714
                                            )
        self.max_initial_weight = self.f_q.max_initial_weight
        self.quantum_prep = Activation_Layer("TanH", self.f_q.encoding_object)
        
        self.i_q = QuantumLayer_MultiQunode(
                                            self.units,
                                            1,
                                            1,
                                            5,
                                            encoding_method="Amplitude_Phase",
                                            regularizer=regularizers.L1(l1=0.01),
                                            max_initial_weight=self.max_initial_weight
                                            )
        self.c_q = QuantumLayer_MultiQunode(
                                            self.units,
                                            1,
                                            1,
                                            5,
                                            encoding_method="Amplitude_Phase",
                                            regularizer=regularizers.L1(l1=0.01),
                                            max_initial_weight=self.max_initial_weight
                                            )
        
        self.o_q = QuantumLayer_MultiQunode(
                                            self.units,
                                            1,
                                            1,
                                            5,
                                            encoding_method="Amplitude_Phase",
                                            regularizer=regularizers.L1(l1=0.01),
                                            max_initial_weight=self.max_initial_weight
                                            )
        
        super(QLSTM, self).__init__(**kwargs)

    def build(self, input_shape):
        self.f = Dense(units=2*self.units,
                       input_shape=[input_shape[-1] + self.units])
        
        self.i = Dense(units=2*self.units,
                       input_shape=[input_shape[-1] + self.units])
        
        self.c = Dense(units=2*self.units,
                       input_shape=[input_shape[-1] + self.units])
        
        self.o = Dense(units=2*self.units,
                       input_shape=[input_shape[-1] + self.units])
        
        self.built = True

    def call(self, inputs, states):
        h,c = states[0], states[1]
        
        x = tf.concat([h,inputs], axis=-1)
        
        y_ft = self.recurrent_activation(self.f_q(self.quantum_prep(self.f(x))))
                

        y_it = self.recurrent_activation(self.i_q(self.quantum_prep(self.i(x))))
        
        y_c = self.recurrent_activation(self.c_q(self.quantum_prep(self.c(x))))
        
        c = c * y_ft + y_it * y_c
        
        y_o = self.recurrent_activation(self.quantum_prep(self.o_q(self.o(x))))
        
        h = y_o * self.activation(c)
        
        output = h
        
        return output, [h,c]