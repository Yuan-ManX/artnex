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
from core import get_array_module


class Optimizer:
    def __init__(self, params_iter, initial_lr=0.03, grad_clip_threshold=None):
        """
        Constructor for the Optimizer class.

        Args:
            params_iter (iterable): An iterable containing model parameters to be optimized.
            initial_lr (float): Initial learning rate.
            grad_clip_threshold (float or None): Threshold for gradient clipping. If None, no clipping is performed.
        """
        self.params = list(params_iter)
        self.lr = initial_lr  # Initialize the learning rate
        self.lr_scheduler = None  # Placeholder for the learning rate scheduler
        self.grad_clip_threshold = grad_clip_threshold  # Threshold for gradient clipping
        self.loss_history = []  # List to store loss values during training
        self.learning_rate_history = []  # List to store learning rate values during training

    def set_lr_scheduler(self, lr_scheduler):
        """
        Set the learning rate scheduler.

        Args:
            lr_scheduler: Learning rate scheduler that determines how the learning rate changes over time.
        """
        self.lr_scheduler = lr_scheduler

    def step(self, inputs, targets):
        """
        Performs an optimization step using a mini-batch of inputs and targets.
        Updates each parameter if it has a gradient. Also updates the learning rate if a scheduler is provided.
        Performs gradient clipping if the threshold is specified.
        Records loss and learning rate values.

        Args:
            inputs: The input data for the mini-batch.
            targets: The target data for the mini-batch.
        """
        if self.lr_scheduler is not None:
            self.lr = self.lr_scheduler.get_lr()  # Update learning rate using the scheduler

        total_loss = 0.0

        # Iterate through mini-batches
        for input_batch, target_batch in zip(inputs, targets):
            # Forward pass and compute gradients
            # (This part may vary based on your specific model and loss function)
            loss = self._compute_loss(input_batch, target_batch)
            total_loss += loss

            # Backward pass and parameter updates
            for param in self.params:
                if param.grad is not None:
                    self._clip_gradients(param.grad)  # Clip gradients if specified
                    self._update(param)

        # Record average loss and learning rate values for the mini-batch
        average_loss = total_loss / len(inputs)
        self.loss_history.append(average_loss)
        self.learning_rate_history.append(self.lr)

    def _clip_gradients(self, gradients):
        """
        Clip gradients if a threshold is specified.

        Args:
            gradients: The gradients to be clipped.
        """
        if self.grad_clip_threshold is not None:
            for i in range(len(gradients)):
                gradients[i] = max(min(gradients[i], self.grad_clip_threshold), -self.grad_clip_threshold)

    def _update(self, param):
        """
        Abstract method for updating a specific parameter.

        This method is meant to be overridden by subclasses to provide specific update rules
        for different optimization algorithms.

        Args:
            param: The parameter to be updated.
        """
        raise NotImplementedError()

    def _compute_loss(self, inputs, targets):
        """
        Compute and return the loss for a mini-batch.

        This method should be implemented in subclasses.

        Args:
            inputs: The input data for the mini-batch.
            targets: The target data for the mini-batch.

        Returns:
            float: The loss for the mini-batch.
        """
        raise NotImplementedError()

    def get_loss_history(self):
        """
        Get the history of recorded loss values during training.

        Returns:
            list: List of loss values.
        """
        return self.loss_history

    def get_learning_rate_history(self):
        """
        Get the history of recorded learning rate values during training.

        Returns:
            list: List of learning rate values.
        """
        return self.learning_rate_history
    
class SGD(Optimizer):
    def __init__(self, params_iter, lr=0.01):
        """
        Construct a Stochastic Gradient Descent (SGD) optimizer.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr (float): Learning rate, controls the step size for parameter updates. Default is 0.01.
        """
        super().__init__(params_iter)
        self.lr = lr

    def _update(self, param):
        """
        Update the parameter using the Stochastic Gradient Descent (SGD) rule.

        Args:
            param: Model parameter to be updated.
        """
        # Update the parameter value by subtracting the learning rate multiplied by the parameter gradient
        param.data -= self.lr * param.grad.data

class LearningRateScheduler:
    def __init__(self, initial_lr=0.01):
        self.lr = initial_lr

    def get_lr(self):
        """
        Get the current learning rate.

        Returns:
            float: Current learning rate.
        """
        return self.lr

    def update_lr(self, new_lr):
        """
        Update the learning rate.

        Args:
            new_lr (float): New learning rate.
        """
        self.lr = new_lr


class SGDLRS(Optimizer):
    def __init__(self, params_iter, lr_scheduler=None):
        """
        Construct an SGD optimizer with learning rate scheduling.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr_scheduler: Learning rate scheduler object. Default is None, meaning no scheduling.
        """
        super().__init__(params_iter)
        self.lr_scheduler = lr_scheduler if lr_scheduler is not None else LearningRateScheduler()

    def set_lr_scheduler(self, lr_scheduler):
        """
        Set the learning rate scheduler.

        Args:
            lr_scheduler: Learning rate scheduler object.
        """
        self.lr_scheduler = lr_scheduler

    def _update(self, param):
        """
        Update the parameter using the Stochastic Gradient Descent (SGD) rule with learning rate scheduling.

        Args:
            param: Model parameter to be updated.
        """
        # Update learning rate using the scheduler
        self.lr = self.lr_scheduler.get_lr()

        # Update the parameter value by subtracting the learning rate multiplied by the parameter gradient
        param.data -= self.lr * param.grad.data

class MomentumSGD(Optimizer):
    def __init__(self, params_iter, lr=0.01, momentum=0.9):
        """
        Construct an optimizer with Stochastic Gradient Descent (SGD) and Momentum.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr (float): Learning rate, controls the step size for parameter updates. Default is 0.01.
            momentum (float): Momentum factor, controls the impact of the previous update. Default is 0.9.
        """
        super().__init__(params_iter)
        self.lr = lr
        self.momentum = momentum
        self.param_state = {}  # Used to store the state (momentum) for each parameter

    def _update(self, param):
        """
        Update the parameter using Stochastic Gradient Descent (SGD) with Momentum.

        Args:
            param: Model parameter to be updated.
        """
        xp = get_array_module(param.data)
        param_id = id(param)

        # Initialize the momentum state for the parameter if not already present
        if param_id not in self.param_state:
            self.param_state[param_id] = xp.zeros_like(param.data)

        # Compute the momentum update
        self.param_state[param_id] = self.momentum * self.param_state[param_id] - self.lr * param.grad.data

        # Apply the update to the parameter
        param.data += self.param_state[param_id]

class WeightDecaySGD(Optimizer):
    def __init__(self, params_iter, lr=0.01, momentum=0.9, weight_decay=0.0001):
        """
        Construct an optimizer with Stochastic Gradient Descent (SGD), Momentum, and Weight Decay.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr (float): Learning rate, controls the step size for parameter updates. Default is 0.01.
            momentum (float): Momentum factor, controls the impact of the previous update. Default is 0.9.
            weight_decay (float): Weight decay factor, controls the impact of weight decay. Default is 0.0001.
        """
        super().__init__(params_iter)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.param_state = {}  # Used to store the state (momentum) for each parameter

    def _update(self, param):
        """
        Update the parameter using Stochastic Gradient Descent (SGD) with Momentum and Weight Decay.

        Args:
            param: Model parameter to be updated.
        """
        xp = get_array_module(param.data)
        param_id = id(param)

        if param_id not in self.param_state:
            self.param_state[param_id] = xp.zeros_like(param.data)

        # Compute the momentum update with added weight decay term
        momentum_update = self.momentum * self.param_state[param_id] - self.lr * (param.grad.data + self.weight_decay * param.data)

        # Apply the update to the parameter
        param.data += momentum_update

        # Update the momentum state
        self.param_state[param_id] = momentum_update

class Adam(Optimizer):
    def __init__(self, params_iter, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Construct an optimizer with Adam algorithm.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr (float): Learning rate, controls the step size for parameter updates. Default is 0.001.
            beta1 (float): Exponential decay rate for the first moment estimates. Default is 0.9.
            beta2 (float): Exponential decay rate for the second moment estimates. Default is 0.999.
            epsilon (float): Small constant to prevent division by zero. Default is 1e-8.
        """
        super().__init__(params_iter)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.timestep = 0
        self.moment1 = [np.zeros_like(param.data) for param in self.params]
        self.moment2 = [np.zeros_like(param.data) for param in self.params]

    def _update(self, param):
        """
        Update the parameter using the Adam algorithm.

        Args:
            param: Model parameter to be updated.
        """
        self.timestep += 1
        beta1_t = self.beta1 ** self.timestep
        beta2_t = self.beta2 ** self.timestep

        grad = param.grad
        self.moment1[param.index] = self.beta1 * self.moment1[param.index] + (1 - self.beta1) * grad
        self.moment2[param.index] = self.beta2 * self.moment2[param.index] + (1 - self.beta2) * grad**2

        m_hat = self.moment1[param.index] / (1 - beta1_t)
        v_hat = self.moment2[param.index] / (1 - beta2_t)

        param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RMSprop(Optimizer):
    def __init__(self, params_iter, lr=0.001, gamma=0.9, epsilon=1e-8):
        """
        Construct an optimizer with RMSprop algorithm.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr (float): Learning rate, controls the step size for parameter updates. Default is 0.001.
            gamma (float): Exponential decay rate for the squared gradients moving average. Default is 0.9.
            epsilon (float): Small constant to prevent division by zero. Default is 1e-8.
        """
        super().__init__(params_iter)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.accumulators = [np.zeros_like(param.data) for param in self.params]

    def _update(self, param):
        """
        Update the parameter using RMSprop algorithm.

        Args:
            param: Model parameter to be updated.
        """
        grad = param.grad.data
        self.accumulators[param.index] = self.gamma * self.accumulators[param.index] + (1 - self.gamma) * grad**2
        param.data -= self.lr * grad / (np.sqrt(self.accumulators[param.index]) + self.epsilon)

class Adagrad(Optimizer):
    def __init__(self, params_iter, lr=0.01, epsilon=1e-8):
        """
        Construct an optimizer with Adagrad algorithm.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr (float): Learning rate, controls the step size for parameter updates. Default is 0.01.
            epsilon (float): Small constant to prevent division by zero. Default is 1e-8.
        """
        super().__init__(params_iter)
        self.lr = lr
        self.epsilon = epsilon
        self.accumulators = [np.zeros_like(param.data) for param in self.params]

    def _update(self, param):
        """
        Update the parameter using Adagrad algorithm.

        Args:
            param: Model parameter to be updated.
        """
        grad = param.grad.data
        self.accumulators[param.index] += grad**2
        param.data -= self.lr * grad / (np.sqrt(self.accumulators[param.index]) + self.epsilon)

class Adadelta(Optimizer):
    def __init__(self, params_iter, rho=0.9, epsilon=1e-8):
        """
        Construct an optimizer with Adadelta algorithm.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            rho (float): Decay rate for the moving averages, controls the decay of historical gradients and parameter updates. Default is 0.9.
            epsilon (float): Small constant to prevent division by zero. Default is 1e-8.
        """
        super().__init__(params_iter)
        self.rho = rho
        self.epsilon = epsilon
        self.accumulators = [np.zeros_like(param.data) for param in self.params]
        self.delta_accumulators = [np.zeros_like(param.data) for param in self.params]

    def _update(self, param):
        """
        Update the parameter using Adadelta algorithm.

        Args:
            param: Model parameter to be updated.
        """
        grad = param.grad.data
        self.accumulators[param.index] = self.rho * self.accumulators[param.index] + (1 - self.rho) * grad**2
        update = -np.sqrt(self.delta_accumulators[param.index] + self.epsilon) / np.sqrt(self.accumulators[param.index] + self.epsilon) * grad
        param.data += update
        self.delta_accumulators[param.index] = self.rho * self.delta_accumulators[param.index] + (1 - self.rho) * update**2

class Rprop(Optimizer):
    def __init__(self, params_iter, lr_pos=1.2, lr_neg=0.5, init_step=0.1, max_step=50.0, min_step=1e-6):
        """
        Construct an optimizer with Rprop algorithm.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr_pos (float): Positive learning rate, controls the step size increase for parameter updates. Default is 1.2.
            lr_neg (float): Negative learning rate, controls the step size decrease for parameter updates. Default is 0.5.
            init_step (float): Initial step size. Default is 0.1.
            max_step (float): Maximum step size. Default is 50.0.
            min_step (float): Minimum step size. Default is 1e-6.
        """
        super().__init__(params_iter)
        self.lr_pos = lr_pos
        self.lr_neg = lr_neg
        self.init_step = init_step
        self.max_step = max_step
        self.min_step = min_step
        self.steps = [np.full_like(param.data, init_step) for param in self.params]
        self.prev_grad = [np.zeros_like(param.data) for param in self.params]

    def _update(self, param):
        """
        Update the parameter using Rprop algorithm.

        Args:
            param: Model parameter to be updated.
        """
        grad = param.grad.data
        prev_grad_sign = np.sign(self.prev_grad[param.index])
        grad_sign = np.sign(grad)

        # Update steps
        self.steps[param.index] *= np.where(grad_sign == prev_grad_sign, self.lr_pos, self.lr_neg)
        self.steps[param.index] = np.clip(self.steps[param.index], self.min_step, self.max_step)

        # Update parameter
        param.data -= self.steps[param.index] * grad

        # Save current gradient for the next iteration
        self.prev_grad[param.index] = grad

class AdamW(Optimizer):
    def __init__(self, params_iter, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        """
        Construct an optimizer with AdamW algorithm.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr (float): Learning rate, controls the step size for parameter updates. Default is 0.001.
            beta1 (float): Exponential decay rate for the first moment estimates. Default is 0.9.
            beta2 (float): Exponential decay rate for the second moment estimates. Default is 0.999.
            epsilon (float): Small constant to prevent division by zero. Default is 1e-8.
            weight_decay (float): Weight decay coefficient. Default is 0.01.
        """
        super().__init__(params_iter)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.timestep = 0
        self.moment1 = [np.zeros_like(param.data) for param in self.params]
        self.moment2 = [np.zeros_like(param.data) for param in self.params]

    def _update(self, param):
        """
        Update the parameter using AdamW algorithm.

        Args:
            param: Model parameter to be updated.
        """
        self.timestep += 1
        beta1_t = self.beta1 ** self.timestep
        beta2_t = self.beta2 ** self.timestep

        grad = param.grad.data
        self.moment1[param.index] = self.beta1 * self.moment1[param.index] + (1 - self.beta1) * grad
        self.moment2[param.index] = self.beta2 * self.moment2[param.index] + (1 - self.beta2) * grad**2

        m_hat = self.moment1[param.index] / (1 - beta1_t)
        v_hat = self.moment2[param.index] / (1 - beta2_t)

        # Weight decay update
        param.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * param.data)

class Nadam(Optimizer):
    def __init__(self, params_iter, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Construct an optimizer with Nadam algorithm.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr (float): Learning rate, controls the step size for parameter updates. Default is 0.001.
            beta1 (float): Exponential decay rate for the first moment estimates. Default is 0.9.
            beta2 (float): Exponential decay rate for the second moment estimates. Default is 0.999.
            epsilon (float): Small constant to prevent division by zero. Default is 1e-8.
        """
        super().__init__(params_iter)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.timestep = 0
        self.moment1 = [np.zeros_like(param.data) for param in self.params]
        self.moment2 = [np.zeros_like(param.data) for param in self.params]

    def _update(self, param):
        """
        Update the parameter using Nadam algorithm.

        Args:
            param: Model parameter to be updated.
        """
        self.timestep += 1
        beta1_t = self.beta1 ** self.timestep
        beta2_t = self.beta2 ** self.timestep

        grad = param.grad.data
        self.moment1[param.index] = self.beta1 * self.moment1[param.index] + (1 - self.beta1) * grad
        self.moment2[param.index] = self.beta2 * self.moment2[param.index] + (1 - self.beta2) * grad**2

        m_hat = self.moment1[param.index] / (1 - beta1_t)
        v_hat = self.moment2[param.index] / (1 - beta2_t)

        # Nesterov update
        m_hat_prime = m_hat / (1 - beta1_t)
        param.data -= self.lr * (self.beta1 * m_hat_prime + (1 - self.beta1) * grad) / (np.sqrt(v_hat) + self.epsilon)

class Adalead(Optimizer):
    def __init__(self, params_iter, lr=0.01, c=0.1):
        """
        Construct an optimizer with Adalead algorithm.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr (float): Initial learning rate, controls the step size for parameter updates. Default is 0.01.
            c (float): Parameter controlling the adaptation of the learning rate. Default is 0.1.
        """
        super().__init__(params_iter)
        self.lr = lr
        self.c = c
        self.prev_loss = None

    def _update(self, param, loss):
        """
        Update the parameter using Adalead algorithm.

        Args:
            param: Model parameter to be updated.
            loss: Current loss value.
        """
        if self.prev_loss is not None:
            grad = param.grad.data
            learning_rate = self.lr / (1 + self.c * np.abs((loss - self.prev_loss) / loss))
            param.data -= learning_rate * grad

        self.prev_loss = loss

class LBFGS(Optimizer):
    def __init__(self, params_iter, lr=0.01, history_size=5):
        """
        Construct an optimizer with L-BFGS algorithm.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr (float): Learning rate, controls the step size for parameter updates. Default is 0.01.
            history_size (int): Size of the history to store recent gradients and parameter changes. Default is 5.
        """
        super().__init__(params_iter)
        self.lr = lr
        self.history_size = history_size
        self.memory = []

    def _update(self, param):
        """
        Update the parameter using L-BFGS algorithm.

        Args:
            param: Model parameter to be updated.
        """
        grad = param.grad.data
        param_vector = param.data.flatten()

        if not self.memory:
            # Initialize memory with the current gradient and parameter vector
            self.memory.append((grad, param_vector))
        else:
            # Keep the history size by removing the oldest entry
            if len(self.memory) == self.history_size:
                self.memory.pop(0)

            # Calculate the difference in gradients and parameter vectors
            grad_diff = grad - self.memory[-1][0]
            param_diff = param_vector - self.memory[-1][1]

            # Update memory with the current gradient and parameter vector
            self.memory.append((grad, param_vector))

            # Perform L-BFGS update
            ro = 1 / np.dot(grad_diff, param_diff)
            q = grad.copy()

            for i in range(len(self.memory) - 1, -1, -1):
                alpha = ro * np.dot(self.memory[i][1] - param_vector, q)
                q -= alpha * self.memory[i][0]

            # Scale the final direction
            direction = ro * q

            # Update parameter vector
            param_vector -= self.lr * direction

            # Reshape and update parameter data
            param.data = param_vector.reshape(param.data.shape)

class YellowFin(Optimizer):
    def __init__(self, params_iter, lr=0.1, mu=0.0, zero_debias=True, beta=0.999):
        """
        Construct an optimizer with YellowFin algorithm.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            lr (float): Initial learning rate. Default is 0.1.
            mu (float): Control parameter for Newton's method. Default is 0.0.
            zero_debias (bool): Whether to enable zero debias. Default is True.
            beta (float): Exponential decay rate for moving averages. Default is 0.999.
        """
        super().__init__(params_iter)
        self.lr = lr
        self.mu = mu
        self.zero_debias = zero_debias
        self.beta = beta
        self.v = {}
        self.m = {}
        self.t = 0

    def _update(self, param):
        """
        Update the parameter using YellowFin algorithm.

        Args:
            param: Model parameter to be updated.
        """
        self.t += 1
        grad = param.grad.data
        name = f"param_{param.index}"

        if name not in self.v:
            self.v[name] = np.zeros_like(param.data)
            self.m[name] = np.zeros_like(param.data)

        v = self.v[name]
        m = self.m[name]

        v = self.beta * v + (1 - self.beta) * grad**2
        m = self.beta * m + (1 - self.beta) * grad

        # Bias correction
        if self.zero_debias:
            v_hat = v / (1 - self.beta**self.t)
            m_hat = m / (1 - self.beta**self.t)
        else:
            v_hat = v
            m_hat = m

        # Compute the effective learning rate
        lr_t = self.lr / (1 + self.mu * self.t)

        # Update the parameter
        param.data -= lr_t * m_hat / (np.sqrt(v_hat) + 1e-6)

class FTRLProximal(Optimizer):
    def __init__(self, params_iter, alpha=0.1, beta=1.0, l1=1.0, l2=1.0):
        """
        Construct an optimizer with FTRL-Proximal algorithm.

        Args:
            params_iter (iterable): Iterable containing model parameters to be optimized.
            alpha (float): Learning rate parameter. Default is 0.1.
            beta (float): Parameter for adjusting the learning rate. Default is 1.0.
            l1 (float): L1 regularization strength. Default is 1.0.
            l2 (float): L2 regularization strength. Default is 1.0.
        """
        super().__init__(params_iter)
        self.alpha = alpha
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.z = {}  # Accumulated squared gradients
        self.n = {}  # Accumulated absolute gradients
        self.t = 0  # Time step

    def _update(self, param):
        """
        Update the parameter using FTRL-Proximal algorithm.

        Args:
            param: Model parameter to be updated.
        """
        self.t += 1
        grad = param.grad.data
        name = f"param_{param.index}"

        if name not in self.z:
            self.z[name] = np.zeros_like(param.data)
            self.n[name] = np.zeros_like(param.data)

        z = self.z[name]
        n = self.n[name]

        # Update accumulated squared gradients
        z += grad**2

        # Update accumulated absolute gradients
        n += np.abs(grad)

        # Compute the effective learning rate
        lr = self.alpha / (1 + self.beta * self.t)

        # Compute the proximal term
        prox = np.where(np.abs(n) <= self.l1, 0, (np.sign(n) * self.l1 - n) / (self.l2 + (self.beta + np.sqrt(z)) / self.alpha))

        # Update the parameter
        param.data = -1 / ((self.beta + np.sqrt(z)) / self.alpha) * (prox + grad * lr)

