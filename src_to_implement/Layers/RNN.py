from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
import numpy as np
from copy import deepcopy


def add_bias_ones(x):
    out = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    return out


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_layer = FullyConnected(input_size + hidden_size,
                                           hidden_size)
        self.tanh = TanH()
        self.output_layer = FullyConnected(hidden_size, output_size)
        self.sigmoid = Sigmoid()
        self._memorize = False
        self.hidden_state = np.zeros(hidden_size)
        self.hidden_error = np.zeros((1, hidden_size))
        self.all_output = np.array([]).reshape(0, self.output_size)
        self.hidden_layer_optimizer = False
        self.output_layer_optimizer = False

    def initialize(self, weights_initializer, bias_initializer):
        self.hidden_layer.initialize(weights_initializer, bias_initializer)
        self.output_layer.initialize(weights_initializer, bias_initializer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, mem):
        self._memorize = mem

    def forward(self, input_tensor):
        # Caution we don't have a batch dimension we have
        # (time, feature_vector)
        # BUT we can enter time sequences in batches!!!
        if not self._memorize:
            self.hidden_state = np.zeros(self.hidden_size)
            self.all_output = np.array([]).reshape(0, self.output_size)

        output = []
        all_hidden_states = []
        all_hidden_and_x = []
        for time_step in input_tensor:
            x = np.concatenate((time_step, self.hidden_state))
            all_hidden_and_x.append(x)
            x = x.reshape((1,) + x.shape)

            # calculate
            x = self.hidden_layer.forward(x)
            self.hidden_state = self.tanh.forward(x)
            x = self.output_layer.forward(self.hidden_state)
            x = self.sigmoid.forward(x)

            # save calculation for next iteration or backward
            self.hidden_state = self.hidden_state.reshape(self.hidden_state.shape[-1])
            all_hidden_states.append(self.hidden_state)
            output.append(x)

        # we need to add the bias ones for the hidden layer manually
        # since we also do the optimization update manually
        self.all_hidden_states = np.array(all_hidden_states)
        self.all_hidden_and_x = add_bias_ones(np.array(all_hidden_and_x))

        output = np.array(output)
        output = output.reshape(output.shape[0], output.shape[2])
        self.all_output = np.append(self.all_output, output, axis=0)
        return output

    def backward(self, error_tensor):
        # self.all_hidden_states.append((1, np.zeros(self.hidden_size)))
        self.hidden_error = np.zeros((1, self.hidden_size))

        output_error = []
        last_hidden_error = []
        last_output_error = []
        self.hidden_gradient = np.zeros(self.hidden_layer.weights.shape)
        self.output_gradient = np.zeros(self.output_layer.weights.shape)
        # roll up from behind
        for i in range(len(error_tensor)-1, -1, -1):
            y = self.all_output[i]
            h = self.all_hidden_states[i]
            hx = self.all_hidden_and_x[i]
            err = error_tensor[i]
            err = err.reshape((1,) + err.shape)

            # calculate error after tanh
            self.sigmoid.y = y
            err = self.sigmoid.backward(err)
            last_output_error.append(err)
            self.output_layer.x = np.append(h, 1).reshape(1, h.size+1)
            err = self.output_layer.backward(err)
            self.output_gradient += self.output_layer.gradient_weights
            err = err + self.hidden_error

            self.tanh.y = h
            err = self.tanh.backward(err)
            last_hidden_error.append(err)
            self.hidden_layer.x = hx.reshape(1, hx.size)
            err = self.hidden_layer.backward(err)
            self.hidden_gradient += self.hidden_layer.gradient_weights

            time_step_err = err[0, 0:self.input_size]
            self.hidden_error = err[0, self.input_size:self.input_size+self.hidden_size]
            self.hidden_error = self.hidden_error.reshape(1, self.hidden_size)
            output_error.append(time_step_err)

        self.last_hidden_error = np.array(last_hidden_error).reshape(len(last_hidden_error), last_hidden_error[0].shape[1])
        self.last_output_error = np.array(last_output_error).reshape(len(last_output_error), last_output_error[0].shape[1])
        output_error.reverse()

        output_error = np.array(output_error)

        self.calculate_update(self.hidden_layer.weights,
                              self.hidden_gradient,
                              self.output_layer.weights,
                              self.output_gradient)
        return output_error

    @property
    def weights(self):
        return self.hidden_layer.weights

    @weights.setter
    def weights(self, w):
        self.hidden_layer.weights = w

    @property
    def gradient_weights(self):
        return self.hidden_gradient

    @property
    def optimizer(self):
        return self.hidden_layer_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self.hidden_layer_optimizer = deepcopy(optimizer)
        self.output_layer_optimizer = deepcopy(optimizer)

    def calculate_update(self,
                         hidden_weight_tensor,
                         hidden_gradient_tensor,
                         output_weight_tensor,
                         output_gradient_tensor):
        if self.hidden_layer_optimizer:
            self.hidden_layer.weights = self.hidden_layer_optimizer.\
                    calculate_update(hidden_weight_tensor,
                                     hidden_gradient_tensor)
        if self.output_layer_optimizer:
            self.output_layer.weights = self.output_layer_optimizer.\
                    calculate_update(output_weight_tensor,
                                     output_gradient_tensor)
