# --------------- Imports ----------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import RNN, LSTMCell
from tensorflow.keras.backend import dot
from tensorflow.keras import activations, initializers, regularizers
from keras.models import Sequential
from keras.layers import Dense, LSTM
from CV_quantum_layers import QuantumLayer_MultiQunode, Activation_Layer
import pennylane as qml
from pennylane import optimize
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds


tf.keras.backend.set_floatx('float32')
#tf.config.list_physical_devices('GPU')
# ------------ QLSTM Layer --------------
class LSTM(tf.keras.layers.Layer):
    def __init__(self, 
                 units,
                 cutoff,
                 q_mem=True,
                 q_nn=True,
                 BS=False,
                 **kwargs):
        self.units = units
        self.state_size = [self.units, self.units]
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('hard_sigmoid')
        self.kernel_initializer = initializers.get('glorot_uniform')
        self.recurrent_initializer = initializers.get('orthogonal')
        self.bias_initializer = initializers.get('zeros')
        self.q_mem = q_mem
        self.q_nn = q_nn
        self.BS = BS
        self.cutoff = cutoff
        if q_mem:
            #Defining memory_update as a keras layer
            self.dev = qml.device('strawberryfields.tf', wires=self.units, cutoff_dim=self.cutoff)
            weight_shapes= {}
            qnode = self.build_qnode()

            self.qlayer = qml.qnn.KerasLayer(qnode,
                                             weight_shapes=weight_shapes,
                                             output_dim=self.units,
                                             trainable=False)
            self.qnode = qnode
        
        self.f_q = QuantumLayer_MultiQunode(
                                            self.units,
                                            1,
                                            1,
                                            cutoff_dim=self.cutoff,
                                            encoding_method="Amplitude_Phase",
                                            regularizer=regularizers.L1(l1=0.01),
                                            max_initial_weight=None
                                            )
        self.max_initial_weight = self.f_q.max_initial_weight
        
        if q_nn:
            self.f_q = QuantumLayer_MultiQunode(
                                                self.units,
                                                1,
                                                1,
                                                cutoff_dim=self.cutoff,
                                                encoding_method="Amplitude_Phase",
                                                regularizer=regularizers.L1(l1=0.01),
                                                max_initial_weight=self.max_initial_weight
                                                )
            self.max_initial_weight = self.f_q.max_initial_weight
            self.quantum_prep = Activation_Layer("TanH", self.f_q.encoding_object)
            self.i_q = QuantumLayer_MultiQunode(
                                                self.units,
                                                1,
                                                1,
                                                cutoff_dim=self.cutoff,
                                                encoding_method="Amplitude_Phase",
                                                regularizer=regularizers.L1(l1=0.01),
                                                max_initial_weight=self.max_initial_weight
                                                )
            self.c_q = QuantumLayer_MultiQunode(
                                                self.units,
                                                1,
                                                1,
                                                cutoff_dim=self.cutoff,
                                                encoding_method="Amplitude_Phase",
                                                regularizer=regularizers.L1(l1=0.01),
                                                max_initial_weight=self.max_initial_weight
                                                )
            self.o_q = QuantumLayer_MultiQunode(
                                                self.units,
                                                1,
                                                1,
                                                cutoff_dim=self.cutoff,
                                                encoding_method="Amplitude_Phase",
                                                regularizer=regularizers.L1(l1=0.01),
                                                max_initial_weight=self.max_initial_weight
                                                )

        super(LSTM, self).__init__(**kwargs)
    def build_qnode(self):

        @qml.qnode(self.dev, interface='tf')
        def qnode(inputs):
            c, y_ft, itC = tf.split(inputs, 3)
            '''SQUEEZE INITIALIZATION'''
            # Use previous expectation values to create a squeezed vacuum state for each qumode
            for i in range(self.units):
                qml.Squeezing(c[i], 0.0, wires=i) 
            if self.BS:
                for i in range(self.units-1):
                    qml.Beamsplitter(np.pi/4, np.pi/2, wires=[i,i+1])

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
            itC = itC * self.max_initial_weight

            for i in range(self.units):
                qml.Displacement(itC[i], 0.0, wires=i)


            '''MEASUREMENT'''

            return [qml.expval(qml.X(i)) for i in range(self.units)]
        return qnode

    def build(self, input_shape):
        scale = 2 if self.q_nn else 1
        self.f = Dense(units=self.units*scale,
                       input_shape=[input_shape[-1] + self.units])
        self.i = Dense(units=self.units*scale,
                       input_shape=[input_shape[-1] + self.units])
        self.c = Dense(units=self.units*scale,
                       input_shape=[input_shape[-1] + self.units])
        self.o = Dense(units=self.units*scale,
                       input_shape=[input_shape[-1] + self.units])
        self.built = True

    def call(self, inputs, states):
        h,c = states[0], states[1]
        x = tf.concat([h,inputs], axis=-1)
        if self.q_nn:
            y_ft = self.recurrent_activation(self.f_q(self.quantum_prep(self.f(x))))

            y_it = self.recurrent_activation(self.i_q(self.quantum_prep(self.i(x))))
            y_c = self.recurrent_activation(self.c_q(self.quantum_prep(self.c(x))))
            y_o = self.recurrent_activation(self.o_q(self.quantum_prep(self.o(x))))
            itC = y_it * y_c
        else:
            y_ft = self.recurrent_activation(self.f(x))

            y_it = self.recurrent_activation(self.i(x))
            y_c = self.recurrent_activation(self.c(x))
            y_o = self.recurrent_activation(self.o(x))
            itC = y_it * y_c
        if self.q_mem:
            '''
            QUANTUM MEMORY SEGMENT:
            '''
            # returns expectation values from quantum memory circuit
            q_inputs = tf.concat([c,y_ft,itC], axis=-1)
            c_list = self.qlayer(q_inputs)
            #c = tf.reshape(c_list, (1, self.units))
            c = c_list
        else:
            c = c * y_ft + itC

        h = y_o * self.activation(c)
        output = h
        return output, [h,c]


# ------------ LOAD DATA -------------
train_data = pd.read_csv('train_data.csv').to_numpy()
test_data = pd.read_csv('test_data.csv').to_numpy()

x_train = train_data[:, 1:101].reshape((1700, 100, 1))
y_train = train_data[:, 101].reshape((1700, 1))

x_test = test_data[:, 1:101].reshape(450, 100, 1)
y_test = test_data[:, 101].reshape(450, 1)


# ----------- SACRED EXPERIMENT ------------
n = 1
RANDOM_SEED = 30

OPTIMIZER = "adam"
LOSS_FUNCTION = "mean_squared_error"
EXPERIMENT_NAME = "QLSTM_full_experiment"
ex = Experiment(EXPERIMENT_NAME)
ex.observers.append(FileStorageObserver(EXPERIMENT_NAME))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.capture
def log_performance(_run, logs, epoch, model):
    """Logs performance with sacred framework"""
    _run.log_scalar("loss", float(logs.get("loss")), epoch)
    _run.log_scalar("accuracy", float(logs.get("accuracy")), epoch)
    _run.log_scalar("epoch", int(epoch), epoch)


class LogPerformance(Callback):
    """Logs performance"""

    def on_batch_end(self, epoch, logs=None):
        """Log key metrics on every 10 batches"""
        if n%10 == 0:
            log_performance(logs=logs, epoch=epoch, model=self.model)  # pylint: disable=E1120


@ex.config
def confnet_config():
    """Default config"""
    BATCH_SIZE = 1
    NUM_EPOCHS = 1
    UNITS = 4
    CUTOFF = 3
    QNN = False
    QMEM = False
    BEAMSPLITTER = True


@ex.automain
def define_and_train(_config):
    """Build and run the network"""
    tf.random.set_seed(RANDOM_SEED)

    # config variables
    units = _config['UNITS']
    cutoff = _config['CUTOFF']
    batch_size = _config['BATCH_SIZE']
    epochs = _config['NUM_EPOCHS']
    q_nn = _config['QNN']
    q_mem = _config['QMEM']
    BS = _config['BEAMSPLITTER']
 

    # Build the LSTM model
    cell = LSTM(units, 
                cutoff=cutoff,
                q_mem=q_mem,
                q_nn=q_nn,
                BS=BS
                )   # first argument n wires, second argument cutoff 
    layer = RNN(cell, return_sequences=False)
    model = Sequential()
    model.add(layer)
    model.add(Dense(35))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, run_eagerly=True)

    # Train the model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=LogPerformance())

