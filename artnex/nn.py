# MIT License

# Copyright (c) 2023 Yuan-Man

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
from core import *


class Layer:
    def __init__(self):
        self.param_names = set()  # Set to store the names of parameters in the layer
        self.cuda = False  # Flag indicating whether the layer is running on GPU, default is False
        self.eval_mode = False  # Flag indicating whether the layer is in evaluation mode, default is training mode

    def __call__(self, *inputs):
        # Convert inputs to Variables and call the _forward method for forward propagation
        inputs = [to_variable(var_or_data, self.cuda) for var_or_data in inputs]
        return self._forward(*inputs)

    def __setattr__(self, name, value):
        # When setting attributes, add them to the param_names set if they are of type Parameter or Layer
        if isinstance(value, Parameter) or isinstance(value, Layer):
            self.param_names.add(name)
        super().__setattr__(name, value)

    def params(self):
        # Generator, returns an iterator over all parameters in the layer
        for name in self.param_names:
            value = getattr(self, name)
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Layer):
                yield from value.params()

    def named_params(self, prefix=''):
        # Generator, returns an iterator over all parameters and their names in the layer, with an optional prefix
        for name in self.param_names:
            value = getattr(self, name)
            fullname = prefix + '.' + name if prefix else name
            if isinstance(value, Parameter):
                yield fullname, value
            else:
                yield from value.named_params(fullname)

    def sublayers(self):
        # Generator, returns an iterator over all sub-layers in the layer
        for name in self.param_names:
            value = getattr(self, name)
            if isinstance(value, Layer):
                yield value
                yield from value.sublayers()

    def zero_grads(self):
        # Zero the gradients of all parameters in the layer
        for param in self.params():
            param.zero_grad()

    def to_cuda(self):
        # Move the layer and all its parameters to GPU
        self.cuda = True
        for p in self.params():
            p.to_cuda()
        for l in self.sublayers():
            l.cuda = True
        return self

    def to_cpu(self):
        # Move the layer and all its parameters to CPU
        self.cuda = False
        for p in self.params():
            p.to_cpu()
        for l in self.sublayers():
            l.cuda = False
        return self

    def save_weights(self, path):
        # Save the weights of all parameters in the layer to a file
        param_dict = {name: to_numpy(param.data) for name, param in self.named_params()}
        np.savez_compressed(path, **param_dict)

    def load_weights(self, path):
        # Load the weights of all parameters in the layer from a file
        param_dict = np.load(path + '.npz')
        for name, param in self.named_params():
            param.data = param_dict[name]
            if self.cuda:
                param.to_cuda()

    def train(self):
        # Set the layer and all its sub-layers to training mode
        self.eval_mode = False
        for l in self.sublayers():
            l.eval_mode = False

    def eval(self):
        # Set the layer and all its sub-layers to evaluation mode
        self.eval_mode = True
        for l in self.sublayers():
            l.eval_mode = True

    def _forward(self, *inputs):
        # Placeholder for the specific implementation of forward propagation, to be implemented by subclasses
        raise NotImplementedError()

class Linear(Layer):
    def __init__(self, in_size, out_size):
        """
        Construct a linear layer.

        Args:
            in_size (int): Number of input features.
            out_size (int): Number of output features.
        """
        super().__init__()

        # Initialize weight matrix W and bias vector b
        # Use the Xavier/Glorot initialization method, which adjusts the initial range of weights based on input and output sizes for more stable training
        self.w = Parameter(np.random.randn(in_size, out_size) / np.sqrt(in_size), name='W')
        self.b = Parameter(np.zeros(out_size), name='b')

    def _forward(self, x):
        """
        Perform forward propagation for the linear layer.

        Args:
            x: Input tensor, shape (batch_size, in_size).

        Returns:
            Output tensor, shape (batch_size, out_size).
        """
        # Linear transformation: x @ w + b
        return x @ self.w + self.b

class Sigmoid(Layer):
    def _forward(self, x):
        """
        Perform forward propagation for the sigmoid activation function.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the sigmoid activation.
        """
        # Apply the sigmoid activation function: 1 / (1 + exp(-x))
        return 1 / (1 + np.exp(-x))
    
class ReLU(Layer):
    def __init__(self):
        """
        Construct the Rectified Linear Unit (ReLU) activation layer.
        """
        super().__init__()

    def _forward(self, x):
        """
        Perform forward propagation of the ReLU activation function.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the ReLU activation function.
        """
        # Apply the ReLU activation function
        return np.maximum(0, x)

class Tanh(Layer):
    def __init__(self):
        """
        Construct the Hyperbolic Tangent (tanh) activation layer.
        """
        super().__init__()

    def _forward(self, x):
        """
        Perform forward propagation of the tanh activation function.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the tanh activation function.
        """
        # Apply the tanh activation function
        # Tanh squashes input values to the range (-1, 1)
        return np.tanh(x)

class Softmax(Layer):
    def __init__(self):
        """
        Construct the Softmax activation layer.
        """
        super().__init__()

    def _forward(self, x):
        """
        Perform forward propagation of the Softmax activation function.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the Softmax activation function.
        """
        # Apply the Softmax activation function
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Subtracting max for numerical stability
        softmax_values = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        return softmax_values

class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        """
        Construct the Leaky ReLU activation layer.

        Args:
            alpha (float): Slope of the negative part of the activation. Default is 0.01.
        """
        super().__init__()
        self.alpha = alpha

    def _forward(self, x):
        """
        Perform forward propagation of the Leaky ReLU activation function.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the Leaky ReLU activation function.
        """
        # Apply the Leaky ReLU activation function
        return np.where(x > 0, x, self.alpha * x)

class PReLU(Layer):
    def __init__(self, alpha=0.01):
        """
        Construct the Parametric ReLU (PReLU) activation layer.

        Args:
            alpha (float): Initial value of the learnable parameter. Default is 0.01.
        """
        super().__init__()
        self.alpha = Parameter(np.full(1, alpha), name='alpha')

    def _forward(self, x):
        """
        Perform forward propagation of the Parametric ReLU (PReLU) activation function.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the PReLU activation function.
        """
        # Apply the PReLU activation function
        return np.where(x > 0, x, self.alpha.data * x)

class L1Regularization(Layer):
    def __init__(self, weight_decay=1e-4):
        """
        Construct a layer for L1 regularization.

        Args:
            weight_decay (float): L1 regularization strength. Default is 1e-4.
        """
        super().__init__()
        self.weight_decay = weight_decay

    def _forward(self, x):
        """
        Perform forward propagation with L1 regularization.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying L1 regularization.
        """
        # Calculate the L1 regularization term and add it to the loss
        l1_loss = self.weight_decay * np.sum(np.abs(x.data))
        self.regularization_loss = l1_loss  # Save the regularization loss for monitoring
        return x

    def get_regularization_loss(self):
        """
        Get the accumulated L1 regularization loss.

        Returns:
            Scalar value representing the L1 regularization loss.
        """
        return self.regularization_loss

class L2Regularization(Layer):
    def __init__(self, weight_decay=1e-4):
        """
        Construct a layer for L2 regularization.

        Args:
            weight_decay (float): L2 regularization strength. Default is 1e-4.
        """
        super().__init__()
        self.weight_decay = weight_decay

    def _forward(self, x):
        """
        Perform forward propagation with L2 regularization.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying L2 regularization.
        """
        # Calculate the L2 regularization term and add it to the loss
        l2_loss = 0.5 * self.weight_decay * np.sum(x.data**2)
        self.regularization_loss = l2_loss  # Save the regularization loss for monitoring
        return x

    def get_regularization_loss(self):
        """
        Get the accumulated L2 regularization loss.

        Returns:
            Scalar value representing the L2 regularization loss.
        """
        return self.regularization_loss

class Dropout(Layer):
    def __init__(self, p=0.5):
        """
        Construct a Dropout layer.

        Args:
            p (float): Probability of dropping out a neuron. Default is 0.5.
        """
        super().__init__()
        self.p = p

    def _forward(self, x):
        """
        Perform forward propagation with dropout.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying dropout during training.
        """
        if self.eval_mode:
            # During evaluation, simply return the input without dropout
            return x
        else:
            xp = get_array(x.data)

            # Generate a binary mask where values are set to 1 with probability (1 - p)
            mask = (xp.random.rand(*x.shape) < 1 - self.p).astype(x.dtype)

            # Scale the input values based on the dropout mask
            # This scales the values by 1/(1 - p) to maintain the expected value during training
            return x * mask / (1 - self.p)

class Softmax1D(Layer):
    def _forward(self, x):
        """
        Perform forward propagation for the 1D softmax activation function.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the 1D softmax activation.
        """
        # Calculate the exponentials of each element in the input
        exp_x = np.exp(x)

        # Calculate the sum along the last axis
        exp_sum = exp_x.sum(axis=-1, keepdims=True)

        # Normalize by dividing each element by the sum
        # This ensures that the output is a valid probability distribution
        softmax_output = exp_x / exp_sum

        return softmax_output

class SoftmaxCrossEntropy1D(Layer):    
    def _forward(self, x, t):
        """
        Perform forward propagation for 1D Softmax Cross Entropy loss.

        Args:
            x: Input tensor.
            t: Target tensor (ground truth labels).

        Returns:
            Scalar value representing the loss.
        """
        # Calculate probabilities using the Softmax activation
        probs = Softmax1D()(x)

        # Ensure numerical stability by applying a small clipping and taking the logarithm
        clipped_probs = Clip(1e-15, 1.0)(probs)
        log_probs = Log()(clipped_probs)

        # Create one-hot encoded labels
        onehots = np.eye(probs.shape[-1])[to_numpy(t.data)]

        # Calculate the cross-entropy loss
        loss = -(log_probs * onehots).sum() / x.shape[0]

        return loss

class StepLR:
    def __init__(self, initial_lr=0.01, decay_factor=0.1, step_size=10):
        """
        Learning rate scheduler using step decay.

        Args:
            initial_lr (float): Initial learning rate. Default is 0.01.
            decay_factor (float): Factor by which the learning rate is multiplied at each step. Default is 0.1.
            step_size (int): Number of epochs after which the learning rate is decayed. Default is 10.
        """
        self.lr = initial_lr
        self.decay_factor = decay_factor
        self.step_size = step_size
        self.current_epoch = 0

    def step(self):
        """
        Perform a learning rate step, updating the learning rate based on the step decay schedule.
        """
        self.current_epoch += 1
        if self.current_epoch % self.step_size == 0:
            self.lr *= self.decay_factor

    def get_lr(self):
        """
        Get the current learning rate.

        Returns:
            float: Current learning rate.
        """
        return self.lr

class BatchNorm(Layer):
    def __init__(self, epsilon=1e-5, momentum=0.9):
        """
        Construct the Batch Normalization layer.

        Args:
            epsilon (float): Small constant to prevent division by zero. Default is 1e-5.
            momentum (float): Momentum for moving average of mean and variance. Default is 0.9.
        """
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.scale = Parameter(np.ones(1), name='scale')
        self.bias = Parameter(np.zeros(1), name='bias')
        self.running_mean = None
        self.running_var = None

    def _forward(self, x):
        """
        Perform forward propagation of the Batch Normalization layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying Batch Normalization.
        """
        if self.eval_mode:
            # In evaluation mode, use running mean and variance
            normalized_x = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        else:
            # In training mode, compute batch mean and variance
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            # Update running mean and variance using momentum
            if self.running_mean is None:
                self.running_mean = batch_mean
                self.running_var = batch_var
            else:
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            # Normalize using batch mean and variance
            normalized_x = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

        # Scale and shift
        output = self.scale.data * normalized_x + self.bias.data
        return output

def im2col(input_data, kernel_size, stride, padding):
    """
    Convert the input image to im2col matrix format.

    Args:
        input_data: Input image, shape (batch_size, channels, height, width).
        kernel_size (tuple): Size of the convolutional kernel, shape (kernel_height, kernel_width).
        stride (int): Convolution stride.
        padding (int): Convolution padding.

    Returns:
        Im2col matrix, shape (batch_size, new_height, new_width, channels, kernel_height, kernel_width).
    """
    batch_size, channels, height, width = input_data.shape
    kernel_height, kernel_width = kernel_size

    # Calculate the output size after convolution
    new_height = (height - kernel_height + 2 * padding) // stride + 1
    new_width = (width - kernel_width + 2 * padding) // stride + 1

    # Use numpy's stride_tricks to generate the im2col matrix
    col = np.lib.stride_tricks.as_strided(input_data,
                                          shape=(batch_size, channels, new_height, new_width, kernel_height, kernel_width),
                                          strides=(input_data.strides[0], input_data.strides[1], stride * input_data.strides[2],
                                                   stride * input_data.strides[3], input_data.strides[2], input_data.strides[3]))

    return col

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Construct a 2D convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel, can be a single integer or a tuple of two integers.
            stride (int): Convolution stride. Default is 1.
            padding (int): Convolution padding. Default is 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize convolutional kernel parameters
        self.weights = Parameter(np.random.randn(out_channels, in_channels, *self.kernel_size) / np.sqrt(in_channels),
                                  name='weights')
        self.bias = Parameter(np.zeros(out_channels), name='bias')

    def _forward(self, x):
        """
        Perform forward propagation of the 2D convolutional layer.

        Args:
            x: Input tensor, shape (batch_size, in_channels, height, width).

        Returns:
            Output tensor, shape (batch_size, out_channels, new_height, new_width).
        """
        batch_size, _, height, width = x.shape
        kernel_height, kernel_width = self.kernel_size

        # Calculate the output size after convolution
        new_height = (height - kernel_height + 2 * self.padding) // self.stride + 1
        new_width = (width - kernel_width + 2 * self.padding) // self.stride + 1

        # Use im2col to transform input data
        col_x = im2col(x, self.kernel_size, self.stride, self.padding)

        # Flatten the weights for matrix multiplication
        flat_weights = self.weights.data.reshape(self.out_channels, -1)

        # Perform convolution
        conv_result = np.dot(flat_weights, col_x) + self.bias.data.reshape(-1, 1)

        # Reshape the result back to the output shape
        output = conv_result.reshape(batch_size, self.out_channels, new_height, new_width)

        return output

def im2col_3d(input_data, kernel_size, stride, padding):
    """
    Convert a 3D input tensor to a 2D matrix for efficient convolution.

    Args:
        input_data: Input tensor with shape (batch_size, channels, depth, height, width).
        kernel_size (tuple): Size of the convolutional kernel (depth, height, width).
        stride (int): Convolutional stride.
        padding (int): Convolutional padding.

    Returns:
        col: 2D matrix representation of the input data.
    """
    batch_size, channels, depth, height, width = input_data.shape
    kernel_depth, kernel_height, kernel_width = kernel_size

    # Calculate the dimensions of the output matrix
    out_depth = (depth - kernel_depth + 2 * padding) // stride + 1
    out_height = (height - kernel_height + 2 * padding) // stride + 1
    out_width = (width - kernel_width + 2 * padding) // stride + 1

    # Apply padding to the input data
    padded_data = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding), (padding, padding)),
                         mode='constant')

    # Initialize an empty matrix to store the transformed data
    col = np.zeros((batch_size, channels, kernel_depth, kernel_height, kernel_width, out_depth, out_height, out_width))

    # Extract patches from the padded data and arrange them in the output matrix
    for d in range(kernel_depth):
        d_max = d + stride * out_depth
        for h in range(kernel_height):
            h_max = h + stride * out_height
            for w in range(kernel_width):
                w_max = w + stride * out_width
                col[:, :, d, h, w, :, :, :] = padded_data[:, :, d:d_max:stride, h:h_max:stride, w:w_max:stride]

    # Reshape the matrix to 2D
    col = col.transpose(0, 5, 6, 7, 1, 2, 3, 4).reshape(batch_size * out_depth * out_height * out_width, -1)

    return col

class Conv3D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Construct a 3D convolutional layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolutional kernel, can be a single integer or a tuple of three integers.
            stride (int): Convolutional stride. Default is 1.
            padding (int): Convolutional padding. Default is 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize convolutional kernel parameters
        self.weights = Parameter(np.random.randn(out_channels, in_channels, *self.kernel_size) / np.sqrt(in_channels),
                                  name='weights')
        self.bias = Parameter(np.zeros(out_channels), name='bias')

    def _forward(self, x):
        """
        Perform forward propagation of the 3D convolutional layer.

        Args:
            x: Input tensor, with shape (batch_size, in_channels, depth, height, width).

        Returns:
            Output tensor, with shape (batch_size, out_channels, new_depth, new_height, new_width).
        """
        batch_size, _, depth, height, width = x.shape
        kernel_depth, kernel_height, kernel_width = self.kernel_size

        # Calculate the output size after convolution
        new_depth = (depth - kernel_depth + 2 * self.padding) // self.stride + 1
        new_height = (height - kernel_height + 2 * self.padding) // self.stride + 1
        new_width = (width - kernel_width + 2 * self.padding) // self.stride + 1

        # Use im2col to transform the input data
        col_x = im2col_3d(x, self.kernel_size, self.stride, self.padding)

        # Flatten the weights for matrix multiplication
        flat_weights = self.weights.data.reshape(self.out_channels, -1)

        # Compute the convolution
        conv_result = np.dot(flat_weights, col_x) + self.bias.data.reshape(-1, 1)

        # Reshape the result back to the output shape
        output = conv_result.reshape(batch_size, self.out_channels, new_depth, new_height, new_width)

        return output

class RNN(Layer):
    def __init__(self, input_size, hidden_size):
        """
        Construct a recurrent layer (RNN).

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize RNN parameters
        self.Wx = Parameter(np.random.randn(hidden_size, input_size), name='Wx')
        self.Wh = Parameter(np.random.randn(hidden_size, hidden_size), name='Wh')
        self.b = Parameter(np.zeros(hidden_size), name='b')

    def _initialize_hidden_state(self, batch_size):
        """
        Initialize the hidden state.

        Args:
            batch_size (int): Size of the current batch.

        Returns:
            Initial hidden state with shape (batch_size, hidden_size).
        """
        return np.zeros((batch_size, self.hidden_size))

    def _forward(self, x):
        """
        Perform forward propagation of the simple RNN.

        Args:
            x: Input tensor with shape (batch_size, sequence_length, input_size).

        Returns:
            Output tensor with shape (batch_size, sequence_length, hidden_size).
        """
        batch_size, sequence_length, _ = x.shape

        # Initialize hidden state
        self.h_t = self._initialize_hidden_state(batch_size)

        # List to store hidden states for each time step
        h_states = []

        # Iterate over the sequence
        for t in range(sequence_length):
            # Current input at time step t
            x_t = x[:, t, :]

            # Compute the hidden state using matrix operations
            self.h_t = np.tanh(self.Wx.data @ x_t.T + self.Wh.data @ self.h_t.T + self.b.data[:, np.newaxis])

            # Append the hidden state to the list
            h_states.append(self.h_t.T)

        # Stack the hidden states along the time axis to get the final output
        output = np.stack(h_states, axis=1)

        return output

class LSTM(Layer):
    def __init__(self, input_size, hidden_size):
        """
        Construct a Long Short-Term Memory (LSTM) layer.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize LSTM parameters
        self.Wxi = Parameter(np.random.randn(hidden_size, input_size), name='Wxi')
        self.Whi = Parameter(np.random.randn(hidden_size, hidden_size), name='Whi')
        self.bi = Parameter(np.zeros(hidden_size), name='bi')

        self.Wxf = Parameter(np.random.randn(hidden_size, input_size), name='Wxf')
        self.Whf = Parameter(np.random.randn(hidden_size, hidden_size), name='Whf')
        self.bf = Parameter(np.zeros(hidden_size), name='bf')

        self.Wxc = Parameter(np.random.randn(hidden_size, input_size), name='Wxc')
        self.Whc = Parameter(np.random.randn(hidden_size, hidden_size), name='Whc')
        self.bc = Parameter(np.zeros(hidden_size), name='bc')

        self.Wxo = Parameter(np.random.randn(hidden_size, input_size), name='Wxo')
        self.Who = Parameter(np.random.randn(hidden_size, hidden_size), name='Who')
        self.bo = Parameter(np.zeros(hidden_size), name='bo')

        # Initialize hidden state and cell state
        self.h_t = None
        self.c_t = None

    def _initialize_hidden_states(self, batch_size):
        """
        Initialize the hidden and cell states.

        Args:
            batch_size (int): Size of the current batch.

        Returns:
            Initial hidden state and cell state, both with shape (batch_size, hidden_size).
        """
        return np.zeros((batch_size, self.hidden_size)), np.zeros((batch_size, self.hidden_size))

    def _sigmoid(self, x):
        """
        Sigmoid activation function.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def _tanh(self, x):
        """
        Hyperbolic tangent activation function.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the hyperbolic tangent activation function.
        """
        return np.tanh(x)

    def _forward(self, x):
        """
        Perform forward propagation of the LSTM.

        Args:
            x: Input tensor with shape (batch_size, sequence_length, input_size).

        Returns:
            Output tensor with shape (batch_size, sequence_length, hidden_size).
        """
        batch_size, sequence_length, _ = x.shape

        # Initialize hidden and cell states
        self.h_t, self.c_t = self._initialize_hidden_states(batch_size)

        # List to store hidden states for each time step
        h_states = []

        # Iterate over the sequence
        for t in range(sequence_length):
            # Current input at time step t
            x_t = x[:, t, :]

            # Forget gate
            f_t = self._sigmoid(self.Wxf.data @ x_t.T + self.Whf.data @ self.h_t.T + self.bf.data[:, np.newaxis])

            # Input gate
            i_t = self._sigmoid(self.Wxi.data @ x_t.T + self.Whi.data @ self.h_t.T + self.bi.data[:, np.newaxis])

            # Cell state
            c_tilde_t = self._tanh(self.Wxc.data @ x_t.T + self.Whc.data @ self.h_t.T + self.bc.data[:, np.newaxis])
            self.c_t = f_t * self.c_t + i_t * c_tilde_t

            # Output gate
            o_t = self._sigmoid(self.Wxo.data @ x_t.T + self.Who.data @ self.h_t.T + self.bo.data[:, np.newaxis])

            # Hidden state
            self.h_t = o_t * self._tanh(self.c_t)

            # Append the hidden state to the list
            h_states.append(self.h_t.T)

        # Stack the hidden states along the time axis to get the final output
        output = np.stack(h_states, axis=1)

        return output

class Attention(Layer):
    def __init__(self, query_size, key_size, value_size):
        """
        Construct an Attention layer.

        Args:
            query_size (int): Size of the query vectors.
            key_size (int): Size of the key vectors.
            value_size (int): Size of the value vectors.
        """
        super().__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size

        # Initialize Attention parameters
        self.Wq = Parameter(np.random.randn(query_size, key_size), name='Wq')
        self.Wk = Parameter(np.random.randn(key_size, key_size), name='Wk')
        self.Wv = Parameter(np.random.randn(value_size, key_size), name='Wv')

        # Softmax layer for attention weights
        self.softmax = Softmax()

    def _scaled_dot_product_attention(self, Q, K, V):
        """
        Scaled Dot-Product Attention.

        Args:
            Q: Query matrix with shape (batch_size, query_size, sequence_length).
            K: Key matrix with shape (batch_size, key_size, sequence_length).
            V: Value matrix with shape (batch_size, value_size, sequence_length).

        Returns:
            Attention-weighted sum of values with shape (batch_size, value_size, sequence_length).
        """
        scale_factor = np.sqrt(K.shape[1])  # Scale factor for better stability
        attention_scores = np.einsum('bik,bjk->bij', Q / scale_factor, K)

        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)

        # Compute attention output using einsum
        attention_output = np.einsum('bij,bjk->bik', attention_weights, V)

        return attention_output

    def _forward(self, Q, K, V):
        """
        Perform forward propagation of the Attention layer.

        Args:
            Q: Query matrix with shape (batch_size, query_size, sequence_length).
            K: Key matrix with shape (batch_size, key_size, sequence_length).
            V: Value matrix with shape (batch_size, value_size, sequence_length).

        Returns:
            Attention-weighted sum of values with shape (batch_size, value_size, sequence_length).
        """
        # Apply linear transformations to the query, key, and value matrices
        Q_transformed = self.Wq.data @ Q
        K_transformed = self.Wk.data @ K
        V_transformed = self.Wv.data @ V

        # Apply Scaled Dot-Product Attention
        attention_output = self._scaled_dot_product_attention(Q_transformed, K_transformed, V_transformed)

        return attention_output

class ResidualConnection(Layer):
    def __init__(self, sublayer, input_size=None):
        """
        Construct a Residual Connection layer.

        Args:
            sublayer (Layer): Sublayer to be connected.
            input_size (int, optional): Size of the input. If None, it will be inferred from the first forward pass.
        """
        super().__init__()
        self.sublayer = sublayer
        self.input_size = input_size
        self.add_parameter('scale', np.ones(1))

    def _forward(self, x):
        """
        Perform forward propagation of the Residual Connection layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the residual connection.
        """
        if self.input_size is None:
            # Infer input size during the first forward pass
            self.input_size = x.shape[-1]

        # Forward pass through the sublayer
        sublayer_output = self.sublayer(x)

        # Check if the input size matches the sublayer output size
        if sublayer_output.shape[-1] != self.input_size:
            raise ValueError("Input size does not match sublayer output size for residual connection.")

        # Apply the residual connection
        output = x + self.scale * sublayer_output

        return output
