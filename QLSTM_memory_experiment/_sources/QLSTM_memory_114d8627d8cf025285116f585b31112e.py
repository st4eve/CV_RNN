# --------------- Imports ----------------
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import RNN, LSTMCell
from tensorflow.keras.backend import dot
from tensorflow.keras import activations, initializers
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os
import pennylane as qml
from pennylane import optimize
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds



tf.keras.backend.set_floatx('float32')
# ------------ QLSTM Layer --------------
class LSTM(tf.keras.layers.Layer):
    def __init__(self, units, cutoff, **kwargs):
        self.units = units
        self.state_size = [self.units, self.units]
        
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('hard_sigmoid')
        self.kernel_initializer = initializers.get('glorot_uniform')
        self.recurrent_initializer = initializers.get('orthogonal')
        self.bias_initializer = initializers.get('zeros')
        
              
        self.cutoff = cutoff
        self.dev = qml.device('strawberryfields.tf', wires=self.units, cutoff_dim=self.cutoff)
        
        
        super(LSTM, self).__init__(**kwargs)
    
    def build_qnode(self):
        @qml.qnode(self.dev, interface='tf')
        def qnode(inputs):
            c = inputs[0:self.units]
            y_ft = inputs[self.units:2*self.units]
            itC = inputs[2*self.units:3*self.units]
            
            
            '''SQUEEZE INITIALIZATION'''
            # Use previous expectation values to create a squeezed vacuum state for each qumode
            for i in range(self.units):
                qml.Squeezing(c[i], 0.0, wires=i) 


            '''ROTATION LAYER'''
            # shape of y_ft: (1,4), values have been hard sigmoided
            # ex. [[0.1, 1, 0.5, 0]]

            #scaling to [0,2π]
            y_ft = 2 * np.pi * y_ft

            for i in range(self.units):
                qml.Rotation(y_ft[i], wires=i)


            '''DISPLACEMENT LAYER'''
            # shape of itC: (1,4), values have been hard sigmoided

            # scale itC to half of cutoff dimension to help with normalization
            itC = itC * self.cutoff / 10

            for i in range(self.units):
                qml.Displacement(itC[i], 0.0, wires=i)


            '''MEASUREMENT'''

            return [qml.expval(qml.X(i)) for i in range(self.units)]
        return qnode

    
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
        
        #Defining memory_update as a keras layer
        weight_shapes= {}
        qnode = self.build_qnode()

        self.qlayer = qml.qnn.KerasLayer(qnode, 
                                         weight_shapes=weight_shapes, 
                                         output_dim=self.units,
                                         trainable=False)
        self.qnode = qnode

    def call(self, inputs, states):
        h,c = states[0], states[1]
        
        x = tf.concat([h,inputs], axis=-1)
        
        y_ft = self.recurrent_activation(dot(x, self.ft) + self.ft_b)
        
        y_it = self.recurrent_activation(dot(x, self.it) + self.it_b)
        
        y_c = self.activation(dot(x, self.c) + self.c_b)
        
        itC = y_it * y_c
        
        
        '''
        QUANTUM MEMORY SEGMENT:
        '''
        # returns expectation values from quantum memory circuit
        q_inputs = tf.concat([c,y_ft,itC], axis=-1)
        c_list = self.qlayer(q_inputs)
        #c = tf.reshape(c_list, (1, self.units))
        c = c_list
        
        y_o = self.recurrent_activation(dot(x, self.o) + self.o_b)
        
        h = y_o * self.activation(c)
        
        output = h
        
        return output, [h,c]


# ------------ LOAD DATA -------------
train_data = pd.read_csv('train_data.csv').to_numpy()
test_data = pd.read_csv('test_data.csv').to_numpy()

x_train = train_data[:, 1:101]
y_train = train_data[:, 101]

x_test = test_data[:, 1:101]
y_test = test_data[:, 101]


# ----------- SACRED EXPERIMENT ------------
i = 1
RANDOM_SEED = 30

OPTIMIZER = "adam"
LOSS_FUNCTION = "mean_squared_error"
EXPERIMENT_NAME = "QLSTM_memory_experiment"
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(FileStorageObserver(EXPERIMENT_NAME))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.capture
def log_performance(_run, logs, epoch, model):
    """Logs performance with sacred framework"""
    _run.log_scalar("loss", float(logs.get("loss")), epoch)
    _run.log_scalar("accuracy", float(logs.get("accuracy")), epoch)
    _run.log_scalar("epoch", int(epoch), epoch)
    #model.save_weights(f"{EXPERIMENT_NAME}/{_run._id}/weights/weight{epoch}.ckpt")  # pylint: disable=W0212


class LogPerformance(Callback):
    """Logs performance"""

    def on_batch_end(self, epoch, logs=None):
        """Log key metrics on every 10 batches"""
        if i%10 == 0:
            log_performance(logs=logs, epoch=epoch, model=self.model)  # pylint: disable=E1120


@ex.config
def confnet_config():
    """Default config"""
    BATCH_SIZE = 1
    NUM_EPOCHS = 1
    UNITS = 4
    CUTOFF = 3


@ex.automain
def define_and_train(batch_size, epochs, units, cutoff):
    """Build and run the network"""
    tf.random.set_seed(RANDOM_SEED)

    # Build the LSTM model
    cell = LSTM(units, cutoff)   # first argument n wires, second argument cutoff 
    layer = RNN(cell, return_sequences=False)
    model = Sequential()
    model.add(layer)
    model.add(Dense(35))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION)

    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

