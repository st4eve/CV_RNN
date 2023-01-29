# %% Imports
import numpy as np
import pennylane as qml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations, layers, models

# %% CV Data Encoding
"""This class wraps around the encoding methods to make them simpler to access.
The conversion factor defines the number of inputs relative to the number of qumodes."""


class CV_Encoding:
    def __init__(self, mode, phase_amplitude=0):
        if mode == "Amplitude":
            self.conversion = 1
        elif mode == "Phase":
            self.conversion = 1
        elif mode == "Amplitude_Phase":
            self.conversion = 2
        else:
            raise ValueError(
                "Please specify a valid encoding type. Valid type are Amplitude, Phase, and Amplitude_Phase."
            )
        self.mode = mode
        self.phase_amplitude = phase_amplitude

    def get_encoding(self, features, wires):
        if self.mode == "Amplitude":
            qml.DisplacementEmbedding(features, wires, method="amplitude", c=0)
        if self.mode == "Phase":
            qml.DisplacementEmbedding(
                features, wires, method="phase", c=self.phase_amplitude
            )
        if self.mode == "Amplitude_Phase":
            for i in range(int(len(features) / 2)):
                qml.Displacement(
                    features[i], features[i + int(len(features) / 2)], wires=wires[i]
                )


# %% CV Measurement
"""This class wraps around the measurement methods to make them simpler to access"""


class CV_Measurement:
    def __init__(self, mode):
        if not (mode == "X_quadrature" or mode == "Fock"):
            raise ValueError(
                "Please specify a valid measurement type. Valid types are X_quadrature and Fock."
            )
        self.mode = mode

    def get_measurement(self, n_qumodes):
        if self.mode == "X_quadrature":
            return [qml.expval(qml.X(wires=i)) for i in range(n_qumodes)]
        if self.mode == "Fock":
            return [qml.probs(wires=i) for i in range(n_qumodes)]


# %% Quantum Activation Layer
class Activation_Layer:
    """This class builds the activation layers that must be placed before the quantum layer.
    Note that the ReLU is a CAPPED ReLU: 1) x<0 -> 0 2) 0<=x<=1 -> x 3) x>1 -> 1."""

    def __init__(self, activation_type, encoding_object):
        self.activation_type = activation_type
        self.encoding_object = encoding_object

    def __call__(self, x):

        # Normalize inputs to [0, 1] range, regarldess of the activation type
        if self.activation_type == "ReLU":  # [0,1]
            x = activations.relu(x, 1)
        if self.activation_type == "Sigmoid":  # [0,1]
            x = activations.sigmoid(x)
        if self.activation_type == "TanH":  # [-1,1]
            x = activations.tanh(x)
            x /= 2
            x += 0.5

        if self.encoding_object.mode == "Amplitude":
            # Displace [0, max_value] with constant phase of 0
            x *= self.encoding_object.phase_amplitude
        if self.encoding_object.mode == "Phase":
            # Displace max_value with phase of [0, 2pi]
            x *= 2 * np.pi
        if self.encoding_object.mode == "Amplitude_Phase":
            # Displace [0, max_value] with phase of [0, 2pi]
            # Split up input to set correct ranges, then join back together
            x_split = list(tf.split(x, 2, axis=1))
            x_split[0] *= self.encoding_object.phase_amplitude
            x_split[1] *= 2 * np.pi
            x = tf.concat([x_split[i] for i in range(2)], axis=1)

        return x


# %% CV Quantum Nodes
def build_cv_quantum_node(
    n_qumodes, cutoff_dim, encoding_object, measurement_object, shots=None
):
    """
    Create CV quantum node. This is the lower level CV object.
    :param n_qumodes: Number of qumodes in the circuit
    :param cutoff_dim: Cutoff dimension to simulate
    :param encoding_object: Encoding object
    :param measurement_object: Measurement object
    :return: CV qnode
    """
    dev = qml.device(
        "strawberryfields.tf", wires=n_qumodes, cutoff_dim=cutoff_dim, shots=shots
    )

    @qml.qnode(dev, interface="tf", shots=shots)
    def cv_nn(
        inputs,
        theta_1,
        phi_1,
        varphi_1,
        r,
        phi_r,
        theta_2,
        phi_2,
        varphi_2,
        a,
        phi_a,
        k,
    ):
        encoding_object.get_encoding(inputs, wires=range(n_qumodes))
        qml.templates.CVNeuralNetLayers(
            theta_1,
            phi_1,
            varphi_1,
            r,
            phi_r,
            theta_2,
            phi_2,
            varphi_2,
            a,
            phi_a,
            k,
            wires=range(n_qumodes),
        )
        return measurement_object.get_measurement(n_qumodes)

    return cv_nn


# %% Full CV Keras Layers
class QuantumLayer_MultiQunode(keras.Model):
    def __init__(
        self,
        n_qumodes,
        n_circuits,
        n_layers,
        cutoff_dim,
        encoding_method="Amplitude_Phase",
        regularizer=None,
        max_initial_weight=None,
        measurement_object=CV_Measurement("X_quadrature"),
        trace_tracking=False,
        shots=None,
        scale_max=1,
    ):
        """
        Initialize Keras NN layer. Example:
        8 inputs coming in from previous layer
        4 qumodes -> n_qumodes = 4
        1 circuit -> n_circuits = 1
        Amplitude & Phase encoding -> encoding_object = CV_Encoding("Amplitude_Phase")
        This results in a single circuit with 4 qumodes because the amplitude+phase encoding
        divides the inputs from the previous layer by 2

        :param n_qumodes: Total number of qumodes in the network
        :param n_circuits: Number of circuits in the network
        :param n_layers: Number of layers within the network
        :param cutoff_dim: Cutoff dimension for the simulation
        :param encoding_object: Encoding method
        :param regularizer: Regularizer object
        :param max_initial_weight: Maximum value allowed for non-phase parameters
        :param measurement_method: Measurement method
        :param trace_tracking: Whether you want traces tracked during training
        """
        super().__init__()
        self.n_circuits = n_circuits
        self.n_qumodes = n_qumodes
        self.n_layers = n_layers
        self.cutoff_dim = cutoff_dim
        self.regularizer = regularizer
        self.encoding_method = encoding_method
        self.measurement_object = measurement_object
        self.trace_tracking = trace_tracking
        self.traces = []
        self.shots = shots
        self.scale_max = 1

        # Calculate number of qumodes based on the down-scaling from encoding and number of circuits
        self.n_qumodes_per_circuit = self.n_qumodes / self.n_circuits
        if self.n_qumodes_per_circuit.is_integer():
            self.n_qumodes_per_circuit = int(self.n_qumodes_per_circuit)
        else:
            raise ValueError(
                "Please ensure the number of inputs divides evenly into the encoding method & number of circuits"
            )

        # If we do not have an max weight, run our max initial weight finder algorithm
        if max_initial_weight is None:
            self.max_initial_weight = self.get_max_non_phase_parameter(
                trace_threshold=0.99
            )
            self.scale_max = scale_max
        else:
            self.max_initial_weight = max_initial_weight

        # Now that we have our max initial weight, we can define our encoding methods and initialize the circuit
        self.encoding_object = CV_Encoding(
            self.encoding_method, self.max_initial_weight
        )
        self.initialize_circuit()

    def initialize_circuit(self):

        # Create the specified number of circuits
        self.circuit_layer = []
        for i in range(self.n_circuits):

            # Make quantum node
            cv_nn = build_cv_quantum_node(
                self.n_qumodes_per_circuit,
                self.cutoff_dim,
                self.encoding_object,
                self.measurement_object,
                shots=self.shots,
            )

            # Define weight shapes
            weight_shapes = self.define_weight_shapes(
                L=self.n_layers, M=self.n_qumodes_per_circuit
            )

            # Define weight specifications
            weight_specs = self.define_weight_specs()

            # Build circuit
            circuit = qml.qnn.KerasLayer(
                cv_nn,
                weight_shapes,
                output_dim=self.n_qumodes_per_circuit,
                weight_specs=weight_specs,
            )

            self.circuit_layer.append(circuit)

    def define_weight_specs(self):
        """
        Define the initial weights and regularizers on each parameter
        :return: dictionary of parameters with their initializers and regularizers
        """
        weight_specs = {
            "theta_1": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)
            },
            "phi_1": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)
            },
            "varphi_1": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)
            },
            "phi_r": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)
            },
            "theta_2": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)
            },
            "phi_2": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)
            },
            "varphi_2": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)
            },
            "phi_a": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)
            },
            "k": {
                "initializer": tf.random_uniform_initializer(minval=0, maxval=2 * np.pi)
            },
            "r": {
                "initializer": tf.keras.initializers.Constant(
                    self.scale_max * self.max_initial_weight
                ),
                "regularizer": self.regularizer,
            },
            "a": {
                "initializer": tf.keras.initializers.Constant(
                    self.scale_max * self.max_initial_weight
                ),
                "regularizer": self.regularizer,
            },
        }
        return weight_specs

    def define_weight_shapes(self, L, M):
        """
        Define the weight shapes for the circuit
        :param L: Number of layers
        :param M: Number of qumodes
        :return: dictionary of parameters with sets of dimensions
        """
        K = int(M * (M - 1) / 2)
        weight_shapes = {
            "theta_1": (L, K),
            "phi_1": (L, K),
            "varphi_1": (L, M),
            "r": (L, M),
            "phi_r": (L, M),
            "theta_2": (L, K),
            "phi_2": (L, K),
            "varphi_2": (L, M),
            "a": (L, M),
            "phi_a": (L, M),
            "k": (L, M),
        }
        return weight_shapes

    def get_traces(self, x):
        # Re-initialize the circuit with Fock measurement
        fock_measurer = QuantumLayer_MultiQunode(
            n_qumodes=self.n_qumodes,
            n_circuits=self.n_circuits,
            n_layers=self.n_layers,
            cutoff_dim=self.cutoff_dim,
            encoding_method=self.encoding_method,
            regularizer=None,
            max_initial_weight=self.max_initial_weight,
            measurement_object=CV_Measurement("Fock"),
        )

        # Split the inputs across the number of circuits
        x_split = list(tf.split(x, self.n_circuits, axis=1))

        # Initialize the model, so the weights can be copied.
        # Without sending one input through, the weights don't get set.
        # Then, initialize the weights, and find the traces.
        for i in range(self.n_circuits):
            x_subsample = x_split[i]
            fock_measurer.circuit_layer[i](x_subsample[0])
            fock_measurer.circuit_layer[i].set_weights(
                self.circuit_layer[i].get_weights()
            )

            for sample in x_subsample:
                fock_dist = fock_measurer.circuit_layer[i](sample)
                average_trace = (
                    np.sum(tf.math.real(fock_dist)) / self.n_qumodes_per_circuit
                )
                self.traces.append(average_trace)

    def get_max_non_phase_parameter(self, trace_threshold=0.99):
        """
        Find the maximum value for the non-phase weights and encoding bounds.
        First, we get an initial guess (upper bound) with a basic displacement gate.
        Then, we create a random set of inputs bounded by our 'max_value' and
        initialize a network using the 'max_value' as the upper bound on the non-phase parameters.
        We run the inputs through this random and check if the resulting trace is within our threshold.
        If not, we lower our 'max_value' and repeat until we get lots of iterations with a good trace.
        You can adjust the count value as needed.
        """
        # Get initial guess by basic max displacement algorithm
        max_value = find_max_displacement(self.cutoff_dim, trace_threshold)

        # Define the value to decrement by. This is a heuristic based on values observed.
        # We could alternatively decrement the value by something like 1% of the current value
        # but that gets slow as the values get smaller.
        decrement = max_value / 200

        count = 0
        while count < 100:

            # Get inputs using our
            if self.encoding_method == "Phase":
                inputs = tf.random.uniform([self.n_qumodes], minval=0, maxval=2 * np.pi)
            if self.encoding_method == "Amplitude":
                inputs = max_value * tf.ones(self.n_qumodes)
            if self.encoding_method == "Amplitude_Phase":
                t1 = max_value * tf.ones(int(self.n_qumodes))
                t2 = tf.random.uniform(
                    [int(self.n_qumodes)], minval=0, maxval=2 * np.pi
                )
                inputs = tf.concat([t1, t2], 0)

            keras_network = QuantumLayer_MultiQunode(
                n_qumodes=self.n_qumodes,
                n_circuits=1,
                n_layers=self.n_layers,
                cutoff_dim=self.cutoff_dim,
                encoding_method=self.encoding_method,
                regularizer=None,
                max_initial_weight=max_value,
                measurement_object=CV_Measurement("Fock"),
            )

            network = keras_network.circuit_layer[0]

            fock_distribution = network(inputs)
            traces = [np.sum(dist) for dist in fock_distribution]
            trace = sum(traces) / self.n_qumodes_per_circuit

            if trace < trace_threshold:
                print("Trace: ", trace, "Value: ", max_value, "Count: ", count)
                max_value -= decrement
                count = 0
            else:
                count += 1

            if max_value <= 0:
                raise Exception(
                    "Max value found to be less than or equal to zero, which is invalid."
                )
                return

        print("max input: ", max_value)
        return max_value

    def call(self, x):
        x_split = list(tf.split(x, self.n_circuits, axis=1))
        output = tf.concat(
            [self.circuit_layer[i](x_split[i]) for i in range(self.n_circuits)], axis=1
        )
        if self.trace_tracking:
            self.get_traces(x)
        return output


#%% Accessory Algorithms
def find_max_displacement(cutoff_dim, trace_threshold):
    """Increase the displacement until the trace treshold is reached"""
    cutoff_dim = int(cutoff_dim)
    dev = qml.device("strawberryfields.tf", wires=1, cutoff_dim=cutoff_dim)

    @qml.qnode(dev, interface="tf")
    def qc(a):
        qml.Displacement(a, 0, wires=0)
        return qml.probs(wires=0)

    a = 0
    trace = 1
    while trace > trace_threshold:
        a += 0.01
        fock_dist = qc(a)
        trace = np.sum(fock_dist)

    return a
