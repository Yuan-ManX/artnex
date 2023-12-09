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
from contextlib import contextmanager

try:
    import cupy as cp
    RUN_CUDA = True
except ImportError:
    RUN_CUDA = False


def to_cupy(data):
    """
    Convert data to a Cupy array if CUDA is available.

    Parameters:
        data: Input data, typically a NumPy array.

    Returns:
        Cupy array representing the input data.

    Raises:
        Exception: If CUDA is not available (RUN_CUDA is False).
    """
    if RUN_CUDA:
        return cp.asarray(data)
    else:
        raise Exception('CUDA is not available')
    
def to_numpy(data):
    """
    Convert data to a NumPy array, considering CUDA availability.

    Parameters:
        data: Input data, which can be a Cupy array.

    Returns:
        NumPy array representing the input data.
    """
    return np.asarray(data) if not RUN_CUDA else cp.asnumpy(data)

def to_variable(data, to_cuda):
    """
    Convert input data to a Variable object.

    Parameters:
        data: Input data, which can be scalar, vector, matrix, etc.
        to_cuda: A boolean indicating whether to move the Variable object to CUDA (GPU).

    Returns:
        Variable object, either wrapping the original input data or a newly created Variable object after conversion.
    """
    # If the input data is already a Variable object, return it directly.
    if isinstance(data, Variable):
        return data
    
    # Create a new Variable object with the input data.
    var = Variable(data)
    
    # If to_cuda is True, move the Variable object to CUDA.
    if to_cuda:
        var.to_cuda()

    return var

def get_array(array):
    """
    Get the array considering CUDA availability.

    Parameters:
        arr: Input array, typically a NumPy or Cupy array.

    Returns:
        The array module to which the input array belongs.
    """
    return np if not RUN_CUDA else cp.get_array_module(array)

class Variable:
    def __init__(self, data, name=None):
        """
        Initialize a Variable object.

        Args:
            data: Input data for the variable.
            name (str): Optional name for the variable.
        """
        if get_array(data) != np:
            self.cuda = True
            self.data = cp.asarray(data)
        else:
            self.cuda = False
            self.data = np.asarray(data)
        self.name = name
        self.grad = None  # Gradient of the variable
        self.func = None  # Associated function in the computation graph
        self.gen = 0  # Generation (used for backward traversal)

    def backward(self):
        """
        Perform backward propagation to compute gradients.
        """
        xp = get_array(self.data)
        self.grad = Variable(xp.ones_like(self.data))  # Gradient initialization

        func_q = [self.func]
        func_set = set(func_q)

        # Backward traversal of the computation graph
        while len(func_q) != 0:
            f = func_q.pop()
            f.backward()

            # Traverse inputs of the function
            for var in f.inputs:
                if var.func is None or var.func in func_set:
                    continue
                func_set.add(var.func)
                func_q.append(var.func)

            # Sort the functions based on generation
            func_q = sorted(func_q, key=lambda f: f.gen)

    def zero_grad(self):
        """
        Set the gradient to zero.
        """
        self.grad = None

    def detach(self):
        """
        Detach the variable from the computation graph.
        """
        self.gen = 0
        self.grad = None
        self.func = None

    def __representation__(self):
        """
        Return a string representation of the variable.
        """
        return str(self.data)

    def __addition__(self, other):
        """
        Overloaded addition operator.

        Args:
            other: Another Variable or numerical value.

        Returns:
            A new Variable representing the sum.
        """
        return Addition()(self, other)

    def __raddition__(self, other):
        """
        Overloaded right addition operator.

        Args:
            other: Another Variable or numerical value.

        Returns:
            A new Variable representing the sum.
        """
        return Addition()(other, self)

    def __subtraction__(self, other):
        """
        Overloaded subtraction operator.

        Args:
            other: Another Variable or numerical value.

        Returns:
            A new Variable representing the difference.
        """
        return Subtraction()(self, other)

    def __rsubtraction__(self, other):
        """
        Overloaded right subtraction operator.

        Args:
            other: Another Variable or numerical value.

        Returns:
            A new Variable representing the difference.
        """
        return Subtraction()(other, self)

    def __multiplication__(self, other):
        """
        Overloaded multiplication operator.

        Args:
            other: Another Variable or numerical value.

        Returns:
            A new Variable representing the product.
        """
        return Multiplication()(self, other)

    def __rmultiplication__(self, other):
        """
        Overloaded right multiplication operator.

        Args:
            other: Another Variable or numerical value.

        Returns:
            A new Variable representing the product.
        """
        return Multiplication()(other, self)

    def __truedivision__(self, other):
        """
        Overloaded true division operator.

        Args:
            other: Another Variable or numerical value.

        Returns:
            A new Variable representing the quotient.
        """
        return Division()(self, other)

    def __rtruedivision__(self, other):
        """
        Overloaded right true division operator.

        Args:
            other: Another Variable or numerical value.

        Returns:
            A new Variable representing the quotient.
        """
        return Division()(other, self)

    def __pow__(self, other):
        """
        Overloaded power operator.

        Args:
            other: Exponent as a Variable or numerical value.

        Returns:
            A new Variable representing the power.
        """
        return Pow(other)(self)

    def __negation__(self):
        """
        Overloaded negation operator.

        Returns:
            A new Variable representing the negation.
        """
        return Negation()(self)

    def __matrimulti__(self, other):
        """
        Overloaded matrix multiplication operator.

        Args:
            other: Another Variable or numerical value.

        Returns:
            A new Variable representing the matrix product.
        """
        return MatrixMultiplication()(self, other)

    def __rmatrimulti__(self, other):
        """
        Overloaded right matrix multiplication operator.

        Args:
            other: Another Variable or numerical value.

        Returns:
            A new Variable representing the matrix product.
        """
        return MatrixMultiplication()(other, self)

    def __getitem__(self, slice):
        """
        Overloaded indexing operator.

        Args:
            slice: Index or slice object.

        Returns:
            A new Variable representing the sliced data.
        """
        return Slice(slice)(self)

    def reshape(self, shape):
        """
        Reshape the Variable.

        Args:
            shape: Desired shape.

        Returns:
            A new Variable with the specified shape.
        """
        return Reshape(shape)(self)

    def transpose(self, axes=None):
        """
        Transpose the Variable.

        Args:
            axes: Optional axes order.

        Returns:
            A new Variable with transposed data.
        """
        return Transpose(axes)(self)

    def sum(self, axes=None, keepdims=False):
        """
        Compute the sum along specified axes.

        Args:
            axes: Axes along which to perform the sum.
            keepdims: Whether to keep the dimensions or not.

        Returns:
            A new Variable representing the summed data.
        """
        return Sum(axes, keepdims)(self)

    def broadcast(self, shape):
        """
        Broadcast the Variable to a new shape.

        Args:
            shape: Target shape.

        Returns:
            A new Variable with broadcasted data.
        """
        return Broadcast(shape)(self)

    def to_cuda(self):
        """
        Move the variable data to CUDA (if available).
        """
        self.cuda = True
        self.data = to_cupy(self.data)
        return self

    def to_cpu(self):
        """
        Move the variable data to CPU.
        """
        self.cuda = False
        self.data = to_numpy(self.data)
        return self

    @property
    def trans(self):
        """
        Transpose property.
        """
        return Transpose(None)(self)

    @property
    def shape(self):
        """
        Return the shape of the variable data.
        """
        return self.data.shape

    @property
    def dtype(self):
        """
        Return the data type of the variable data.
        """
        return self.data.dtype

class Parameter(Variable):
    pass 

NO_GRAD = False

@contextmanager
def no_grad():
    """
    Context manager to temporarily disable gradient tracking during inference.
    """
    global NO_GRAD
    NO_GRAD = True
    yield
    NO_GRAD = False

class Function:
    def __init__(self):
        """
        Initialize the Function object.
        """
        self.gen = 0  # Maximum generation among inputs
        self.inputs = None
        self.outputs = None

    @property
    def to_cuda(self):
        """
        Check inputs for the presence of numpy or cupy Variables.

        Returns:
            bool: True if cupy Variable is present, False otherwise.
        """
        run_np, run_cuda = False, False
        for var_or_data in self.inputs:
            if isinstance(var_or_data, Variable):
                if var_or_data.cuda:
                    run_cuda = True
                else:
                    run_np = True
        if run_np and run_cuda:
            raise Exception('Function inputs have both numpy and cupy Variables, please check!')
        return run_cuda

    def forward(self, *inputs):
        """
        Perform the forward pass of the function.

        Args:
            *inputs: Variable objects or data.

        Returns:
            Variable or tuple of Variables: Output of the forward pass.
        """
        to_cuda = self.to_cuda
        inputs = [to_variable(var_or_data, to_cuda) for var_or_data in inputs]
        outputs = self._forward(*[var.data for var in inputs])
        if not isinstance(outputs, tuple):
            outputs = outputs,

        outputs = [Variable(data) for data in outputs]
        self.gen = max([var.gen for var in inputs])
        for var in outputs:
            var.func = self
            var.gen = self.gen + 1

        # Store inputs & outputs for backward
        if not NO_GRAD:
            self.inputs = inputs
            self.outputs = outputs
        return outputs[0] if len(outputs) == 1 else outputs
    
    __call__ = forward

    def backward(self):
        """
        Perform the backward pass to compute gradients.
        """
        output_grads = [var.grad for var in self.outputs]
        input_grads = self._backward(*output_grads)
        if not isinstance(input_grads, tuple):
            input_grads = input_grads,
        for i, input_grad in enumerate(input_grads):
            if self.inputs[i].grad is None:
                self.inputs[i].grad = input_grad
            else:
                self.inputs[i].grad += input_grad
        for var in self.outputs:
            var.grad = None  # Clear memory & prepare for double backward

    # Override these methods in subclasses
    def _forward(self, *inputs):
        """
        Perform the actual computation in the forward pass.

        Args:
            *inputs: Numpy arrays representing input data.

        Returns:
            Numpy array or tuple of arrays: Output of the computation.
        """
        raise NotImplementedError()

    def _backward(self, *grad):
        """
        Compute the gradients in the backward pass.

        Args:
            *grad: Variable objects representing gradients of the outputs.

        Returns:
            Variable or tuple of Variables: Gradients of the inputs.
        """
        raise NotImplementedError()

class Broadcast(Function):
    def __init__(self, output_shape):
        """
        Initialize the Broadcast object.

        Args:
            output_shape: The desired output shape after broadcasting.
        """
        self.output_shape = output_shape 
        self.x_shape = None

    def _forward(self, x):
        """
        Perform broadcasting in the forward pass.

        Args:
            x: Numpy array representing the input.

        Returns:
            Numpy array: Result of broadcasting.
        """
        xp = get_array(x)   # CUDA compatibility
        self.x_shape = x.shape

        try:
            result = xp.broadcast_to(x, self.output_shape)
        except ValueError as e:
            raise ValueError(f"Broadcasting failed: {e}")

        return result

    def _backward(self, grad):
        """
        Perform debroadcasting in the backward pass.

        Args:
            grad: Numpy array representing the gradient.

        Returns:
            Numpy array: Result of debroadcasting.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        return DeBroadcast(self.x_shape)(grad)

class DeBroadcast(Function):
    def __init__(self, output_shape):
        """
        Initialize the DeBroadcast object.

        Args:
            output_shape: The desired output shape after debroadcasting.
        """
        self.output_shape = output_shape
        self.x_shape = None

    def _forward(self, x):
        """
        Perform debroadcasting in the forward pass.

        Args:
            x: Numpy array representing the input.

        Returns:
            Numpy array: Result of debroadcasting.
        """
        xp = get_array(x)  # CUDA compatibility

        self.x_shape = x.shape
        prefix_ndim = len(x.shape) - len(self.output_shape)
        dims = [prefix_ndim + i for i, (a, b) in enumerate(zip(self.output_shape, x.shape[prefix_ndim:])) if a != b]
        prefix_dims = list(range(prefix_ndim))

        try:
            output = xp.sum(x, axis=tuple(prefix_dims + dims), keepdims=True)
            result = xp.squeeze(output, axis=tuple(prefix_dims))
        except ValueError as e:
            raise ValueError(f"Debroadcasting failed: {e}")

        return result

    def _backward(self, grad):
        """
        Perform broadcasting in the backward pass.

        Args:
            grad: Numpy array representing the gradient.

        Returns:
            Numpy array: Result of broadcasting.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        return Broadcast(self.x_shape)(grad)

class Reshape(Function):
    def __init__(self, target_shape):
        """
        Initialize the Reshape object.

        Args:
            target_shape: The desired target shape after reshaping.
        """
        self.target_shape = target_shape
        self.original_shape = None

    def _forward(self, input_array):
        """
        Perform reshaping in the forward pass.

        Args:
            input_array: Numpy array representing the input.

        Returns:
            Numpy array: Result of the reshaping operation.
        """
        xp = get_array(input_array)   # CUDA compatibility
        self.original_shape = input_array.shape

        try:
            result = xp.reshape(input_array, self.target_shape)
        except ValueError as e:
            raise ValueError(f"Reshaping failed: {e}")

        return result

    def _backward(self, grad):
        """
        Undo the reshaping in the backward pass.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original shape.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        try:
            result = Reshape(self.original_shape)(grad)
        except ValueError as e:
            raise ValueError(f"Undoing reshaping failed: {e}")

        return result
    
class Transpose(Function):
    def __init__(self, axes):
        """
        Initialize the Transpose object.

        Args:
            axes: The desired axes after transposing.
        """
        self.axes = axes

    def _forward(self, input_array):
        """
        Perform transposing in the forward pass.

        Args:
            input_array: Numpy array representing the input.

        Returns:
            Numpy array: Result of the transposing operation.
        """
        xp = get_array(input_array)   # CUDA compatibility

        try:
            result = xp.transpose(input_array, self.axes)
        except ValueError as e:
            raise ValueError(f"Transposing failed: {e}")

        return result

    def _backward(self, grad):
        """
        Undo the transposing in the backward pass.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original axes.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        try:
            result = Transpose(self.axes)(grad)
        except ValueError as e:
            raise ValueError(f"Undoing transposing failed: {e}")

        return result

class Addition(Function):
    def _forward(self, a, b):
        """
        Perform the addition operation in the forward pass.

        Args:
            a: Numpy array representing the first input.
            b: Numpy array representing the second input.

        Returns:
            Numpy array: Result of the addition.
        """
        xp = get_array(a)   # Assuming a and b have the same type
        return xp.add(a, b)

    def _backward(self, grad):
        """
        Compute the gradients in the backward pass for the addition operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            tuple of Variable objects: Gradients of the inputs.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        a_grad, b_grad = self._handle_broadcast(grad)
        return a_grad, b_grad

    def _handle_broadcast(self, grad):
        """
        Handle broadcasting in the backward pass.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            tuple of Variable objects: Gradients of the inputs.
        """
        a_shape, b_shape = self.inputs[0].shape, self.inputs[1].shape

        try:
            a_grad = grad if a_shape == grad.shape else DeBroadcast(a_shape)(grad)
            b_grad = grad if b_shape == grad.shape else DeBroadcast(b_shape)(grad)
        except ValueError as e:
            raise ValueError(f"Handling broadcasting failed: {e}")

        return a_grad, b_grad

class Subtraction(Function):
    def _forward(self, a, b):
        """
        Perform the subtraction operation in the forward pass.

        Args:
            a: Numpy array representing the minuend.
            b: Numpy array representing the subtrahend.

        Returns:
            Numpy array: Result of the subtraction.
        """
        xp = get_array(a)   # Assuming a and b have the same type
        return xp.subtract(a, b)
    
    def _backward(self, grad):
        """
        Compute the gradients in the backward pass for the subtraction operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            tuple of Variable objects: Gradients of the inputs.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        a_grad, b_grad = grad * 1, grad * -1

        # Check and handle broadcasting in the backward pass
        if self.inputs[0].shape != grad.shape:
            a_grad = DeBroadcast(self.inputs[0].shape)(grad)
        if self.inputs[1].shape != grad.shape:
            b_grad = DeBroadcast(self.inputs[1].shape)(grad)

        return a_grad, b_grad

class Multiplication(Function):
    def _forward(self, a, b):
        """
        Perform the multiplication operation in the forward pass.

        Args:
            a: Numpy array representing the first factor.
            b: Numpy array representing the second factor.

        Returns:
            Numpy array: Result of the multiplication.
        """
        xp = get_array(a)   # Assuming a and b have the same type
        return xp.multiply(a, b)

    def _backward(self, grad):
        """
        Compute the gradients in the backward pass for the multiplication operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            tuple of Variable objects: Gradients of the inputs.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        a_grad = self.inputs[1] * grad
        b_grad = self.inputs[0] * grad

        # Check and handle broadcasting in the backward pass
        if self.inputs[0].shape != grad.shape:
            a_grad = DeBroadcast(self.inputs[0].shape)(a_grad)
        if self.inputs[1].shape != grad.shape:
            b_grad = DeBroadcast(self.inputs[1].shape)(b_grad)

        return a_grad, b_grad

class Division(Function):
    def _forward(self, numerator, denominator):
        """
        Perform the division operation in the forward pass.

        Args:
            numerator: Numpy array representing the numerator.
            denominator: Numpy array representing the denominator.

        Returns:
            Numpy array: Result of the division.
        """
        xp = get_array(numerator)   # Assuming numerator and denominator have the same type
        return xp.divide(numerator, denominator)
    
    def _backward(self, grad):
        """
        Compute the gradients in the backward pass for the division operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            tuple of Variable objects: Gradients of the inputs.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        a_grad = grad * 1 / self.inputs[1]
        b_grad = -1 * grad * self.inputs[0] / (self.inputs[1] ** 2)

        # Check and handle broadcasting in the backward pass
        if self.inputs[0].shape != grad.shape:
            a_grad = DeBroadcast(self.inputs[0].shape)(a_grad)
        if self.inputs[1].shape != grad.shape:
            b_grad = DeBroadcast(self.inputs[1].shape)(b_grad)

        return a_grad, b_grad

class Pow(Function):
    def __init__(self, exponent):
        """
        Initialize the Pow object.

        Args:
            exponent: The exponent for the power operation.
        """
        self.exponent = exponent  # int

    def _forward(self, base):
        """
        Perform the power operation in the forward pass.

        Args:
            base: Numpy array representing the base.

        Returns:
            Numpy array: Result of the power operation.
        """
        xp = get_array(base)   # CUDA compatibility
        return xp.power(base, self.exponent)

    def _backward(self, grad):
        """
        Compute the gradients in the backward pass for the power operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient of the input.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        return grad * self.exponent * self.outputs[0] / self.inputs[0]

class Negation(Function):
    def _forward(self, x):
        """
        Perform the negation operation in the forward pass.

        Args:
            x: Numpy array representing the input.

        Returns:
            Numpy array: Result of the negation.
        """
        xp = get_array(x)   # Assuming x has the same type as the result
        return xp.negative(x)

    def _backward(self, grad):
        """
        Compute the gradients in the backward pass for the negation operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient of the input.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        return grad * -1

class Sum(Function):
    def __init__(self, axes, keepdims):
        """
        Initialize the Sum object.

        Args:
            axes: The axes along which the summation is performed.
            keepdims: Whether to keep the dimensions after summation.
        """
        self.axes = axes
        self.keepdims = keepdims
    
    def _forward(self, input_array):
        """
        Perform the summation operation in the forward pass.

        Args:
            input_array: Numpy array representing the input.

        Returns:
            Numpy array: Result of the summation operation.
        """
        xp = get_array(input_array)   # Assuming input_array has the same type as the result
        self.original_shape = input_array.shape
        return xp.sum(input_array, axis=self.axes, keepdims=self.keepdims)

    def _backward(self, grad):
        """
        Compute the gradients in the backward pass for the summation operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original input shape.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        grad_shape = list(grad.data.shape)

        if len(grad_shape) != len(self.original_shape): 
            axes = list(range(len(self.original_shape))) if self.axes is None else self.axes
            for idim in axes:  
                grad_shape.insert(idim, 1)
        
        grad_reshaped = grad.reshape(grad_shape)  
        grad_broadcasted = grad_reshaped.broadcast(self.original_shape)  
        
        return grad_broadcasted

class MatrixMultiplication(Function):
    def _forward(self, matrix_a, matrix_b):
        """
        Perform matrix multiplication in the forward pass.

        Args:
            matrix_a: Numpy array representing the first matrix.
            matrix_b: Numpy array representing the second matrix.

        Returns:
            Numpy array: Result of the matrix multiplication.
        """
        xp = get_array(matrix_a)   # Assuming matrix_a and matrix_b have the same type
        return xp.dot(matrix_a, matrix_b)

    def _backward(self, grad):
        """
        Compute the gradients in the backward pass for matrix multiplication.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            tuple of Variable objects: Gradients with respect to the original matrices.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        xp = get_array(grad)   

        grad_matrix_a = MatrixMultiplication()(grad, self.inputs[1].transpose())
        grad_matrix_b = MatrixMultiplication()(self.inputs[0].transpose(), grad)

        return grad_matrix_a, grad_matrix_b

class Exponential(Function):
    def _forward(self, input_array):
        """
        Perform the exponential operation in the forward pass.

        Args:
            input_array: Numpy array representing the input.

        Returns:
            Numpy array: Result of the exponential operation.
        """
        xp = get_array(input_array)  
        return xp.exp(input_array)
    
    def _backward(self, grad):
        """
        Compute the gradient in the backward pass for the exponential operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original input.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        xp = get_array(grad)  

        return self.outputs[0] * grad

class Slice(Function):
    def __init__(self, slice_obj):
        """
        Initialize the Slice object.

        Args:
            slice_obj: The slice object specifying the slicing operation.
        """
        self.slice_obj = slice_obj

    def _forward(self, input_array):
        """
        Perform slicing in the forward pass.

        Args:
            input_array: Numpy array representing the input.

        Returns:
            Numpy array: Result of the slicing operation.
        """
        self.input_shape = input_array.shape
        return input_array[self.slice_obj]
    
    def _backward(self, grad):
        """
        Compute the gradient in the backward pass for the slicing operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original input shape.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        return SliceGrad(self.input_shape, self.slice_obj)(grad)

class SliceGrad(Function):
    def __init__(self, input_shape, slice_obj):
        """
        Initialize the SliceGrad object.

        Args:
            input_shape: The shape of the original input before slicing.
            slice_obj: The slice object specifying the slicing operation.
        """
        self.input_shape = input_shape
        self.slice_obj = slice_obj
    
    def _forward(self, grad):
        """
        Compute the gradient in the forward pass for the slicing operation.

        Args:
            grad: Numpy array representing the gradient of the output.

        Returns:
            Numpy array: Gradient with respect to the original input shape.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        xp = get_array(grad)   # Assuming grad has the same type as the result
        grad_x = xp.zeros(self.input_shape)
        xp.add.at(grad_x, self.slice_obj, grad)
        return grad_x

    def _backward(self, grad):
        """
        Perform slicing in the backward pass.

        Args:
            grad: Numpy array representing the gradient of the output.

        Returns:
            Numpy array: Result of the slicing operation.
        """
        return Slice(self.slice_obj)(grad)

class Logarithm(Function):
    def _forward(self, input_array):
        """
        Perform the logarithm operation in the forward pass.

        Args:
            input_array: Numpy array representing the input.

        Returns:
            Numpy array: Result of the logarithm operation.
        """
        xp = get_array(input_array) 
        return xp.log(input_array)
    
    def _backward(self, grad):
        """
        Compute the gradient in the backward pass for the logarithm operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original input.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        return grad / self.inputs[0]

class Clip(Function):
    def __init__(self, x_min, x_max):
        """
        Initialize the Clip object.

        Args:
            x_min: Minimum value for clipping.
            x_max: Maximum value for clipping.
        """
        self.x_min = x_min
        self.x_max = x_max 

    def _forward(self, input_array):
        """
        Perform the clip operation in the forward pass.

        Args:
            input_array: Numpy array representing the input.

        Returns:
            Numpy array: Result of the clip operation.
        """
        xp = get_array(input_array) 
        return xp.clip(input_array, self.x_min, self.x_max)
    
    def _backward(self, grad):
        """
        Compute the gradient in the backward pass for the clip operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original input.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        xp = get_array(grad.data) 

        within_bounds = (self.inputs[0].data >= self.x_min) * (self.inputs[0].data <= self.x_max)
        return grad * within_bounds.astype(xp.uint8)

class Relu(Function):
    def _forward(self, input_array):
        """
        Perform the ReLU activation function in the forward pass.

        Args:
            input_array: Numpy array representing the input.

        Returns:
            Numpy array: Result of the ReLU activation function.
        """
        xp = get_array(input_array) 
        return xp.maximum(input_array, xp.zeros_like(input_array))
    
    def _backward(self, grad):
        """
        Compute the gradient in the backward pass for the ReLU activation function.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original input.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        x_grad = grad * (self.inputs[0].data > 0)
        return x_grad
    
class Sigmoid(Function):
    def __init__(self, threshold=0.5):
        """
        Initialize the Sigmoid object.

        Args:
            threshold (float): Threshold value to set output to zero if below.
        """
        self.threshold = threshold
    
    def _forward(self, input_array):
        """
        Perform the modified Sigmoid activation function in the forward pass.

        Args:
            input_array: Numpy array representing the input.

        Returns:
            Numpy array: Result of the modified Sigmoid activation function.
        """
        xp = get_array(input_array)  
        sigmoid_output = 1 / (1 + xp.exp(-input_array))
        return xp.where(sigmoid_output < self.threshold, 0, sigmoid_output)
    
    def _backward(self, grad):
        """
        Compute the gradient in the backward pass for the modified Sigmoid activation function.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original input.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")
        
        sigmoid_output = self.outputs[0]
        grad_input = grad * sigmoid_output * (1 - sigmoid_output)

        return grad_input

class Tanh(Function):
    def __init__(self, scaling_factor=1.0):
        """
        Initialize the Tanh object.

        Args:
            scaling_factor (float): Scaling factor to apply to the Tanh output.
        """
        self.scaling_factor = scaling_factor
    
    def _forward(self, input_array):
        """
        Perform the modified Tanh activation function in the forward pass.

        Args:
            input_array: Numpy array representing the input.

        Returns:
            Numpy array: Result of the modified Tanh activation function.
        """
        xp = get_array(input_array)   
        tanh_output = xp.tanh(input_array)
        return self.scaling_factor * tanh_output
    
    def _backward(self, grad):
        """
        Compute the gradient in the backward pass for the modified Tanh activation function.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original input.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")
        
        tanh_output = self.outputs[0]
        grad_input = grad * self.scaling_factor * (1 - tanh_output**2)

        return grad_input
    
class LeakyReLU(Function):
    def __init__(self, alpha=0.01):
        """
        Initialize the LeakyReLU object.

        Args:
            alpha (float): Leaky parameter for negative input values.
        """
        self.alpha = alpha
    
    def _forward(self, input_array):
        """
        Perform the modified Leaky ReLU activation function in the forward pass.

        Args:
            input_array: Numpy array representing the input.

        Returns:
            Numpy array: Result of the modified Leaky ReLU activation function.
        """
        xp = get_array(input_array)  
        leaky_relu_output = xp.maximum(self.alpha * input_array, input_array)
        return leaky_relu_output
    
    def _backward(self, grad):
        """
        Compute the gradient in the backward pass for the modified Leaky ReLU activation function.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original input.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")
        
        input_array = self.inputs[0].data
        grad_input = grad * ((input_array > 0) + (self.alpha * (input_array <= 0)))

        return grad_input

class ELU(Function):
    def __init__(self, alpha=1.0):
        """
        Initialize the ELU object.

        Args:
            alpha (float): ELU parameter for negative input values.
        """
        self.alpha = alpha
    
    def _forward(self, input_array):
        """
        Perform the modified Exponential Linear Unit (ELU) activation function in the forward pass.

        Args:
            input_array: Numpy array representing the input.

        Returns:
            Numpy array: Result of the modified ELU activation function.
        """
        xp = get_array(input_array)  
        elu_output = xp.where(input_array > 0, input_array, self.alpha * (xp.exp(input_array) - 1))
        return elu_output
    
    def _backward(self, grad):
        """
        Compute the gradient in the backward pass for the modified ELU activation function.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original input.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")
        
        input_array = self.inputs[0].data
        xp = get_array(input_array)

        exp_term = xp.exp(input_array)
        grad_input = grad * xp.where(input_array > 0, 1, self.alpha * exp_term)

        return grad_input

class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        """
        Initialize the Max object.

        Args:
            axis: Axis or axes along which the max operation is performed.
            keepdims: If True, the original dimensions are kept in the result.
        """
        self.axis = axis
        self.keepdims = keepdims

    def _forward(self, x):
        """
        Perform the max operation along specified axes in the forward pass.

        Args:
            x: Numpy array representing the input.

        Returns:
            Numpy array: Result of the max operation.
        """
        xp = get_array(x)  
        return xp.max(x, axis=self.axis, keepdims=self.keepdims)
    
    def _backward(self, grad):
        """
        Compute the gradient in the backward pass for the max operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Variable object: Gradient with respect to the original input.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        x = self.inputs[0]
        xp = get_array(x.data)

        axis = self.axis
        if axis is None:
            axis = list(range(len(x.shape)))
        elif not isinstance(axis, (tuple, list)):
            axis = (self.axis,)
        axis = [ax if ax >= 0 else len(x.shape) + ax for ax in axis]

        grad_shape = [1 if ax in axis else size for ax, size in enumerate(x.shape)]

        max_indices = xp.argmax(x.data, axis=self.axis, keepdims=self.keepdims)
        max_mask = xp.zeros_like(x.data, dtype=xp.uint8)
        max_mask[tuple(max_indices)] = 1

        return grad.reshape(grad_shape) * max_mask

class Concat(Function):
    def __init__(self, axis):
        """
        Initialize the Concat object.

        Args:
            axis: Axis along which the concatenation is performed.
        """
        self.axis = axis
    
    def _forward(self, *xs):
        """
        Perform the concatenation operation along a specified axis in the forward pass.

        Args:
            *xs: Tuple of Numpy arrays representing the input.

        Returns:
            Numpy array: Result of the concatenation operation.
        """
        xp = get_array(xs[0])  
        return xp.concatenate(xs, axis=self.axis)

    def _backward(self, grad):
        """
        Compute the gradient in the backward pass for the concatenation operation.

        Args:
            grad: Variable object representing the gradient of the output.

        Returns:
            Tuple of Variable objects: Gradients with respect to the original inputs.
        """
        if not isinstance(grad, np.ndarray) and not isinstance(grad, cp.ndarray):
            raise TypeError("Input gradient must be a NumPy or Cupy array.")

        slices = []
        start = 0
        for x in self.inputs:
            end = start + x.shape[self.axis]
            slices.append(slice(start, end))
            start = end

        grad_slices = [grad[slices[i]] for i in range(len(self.inputs))]
        return tuple(grad_slices)

