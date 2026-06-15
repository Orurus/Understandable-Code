"""Readable tensor-forge helpers for UnderstandableCode.

This module provides a compact local tensor toolkit with its own vocabulary:
- tensor creation helpers
- parameter wrappers
- module composition
- linear layers
- sequential containers
- SGD and Adam optimizers
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from undml.tensor.tensor_matrix_operation import Tensor


def create_forge_tensor_from_plain_values(values, requires_grad: bool = False, name: str = "tensor"):
    return Tensor(values, requires_grad=requires_grad, name=name)


def create_forge_zero_tensor(shape: Sequence[int], requires_grad: bool = False, name: str = "zeros"):
    from undml.tensor.tensor_matrix_operation import new_tensor_with_zeros

    return Tensor(new_tensor_with_zeros(shape), requires_grad=requires_grad, name=name)


def create_forge_one_tensor(shape: Sequence[int], requires_grad: bool = False, name: str = "ones"):
    from undml.tensor.tensor_matrix_operation import new_tensor_with_ones

    return Tensor(new_tensor_with_ones(shape), requires_grad=requires_grad, name=name)


def create_forge_random_tensor(shape: Sequence[int], minimum: float = -0.1, maximum: float = 0.1, requires_grad: bool = False, name: str = "random"):
    from undml.tensor.tensor_matrix_operation import new_tensor_with_random_values

    return Tensor(new_tensor_with_random_values(shape, minimum, maximum), requires_grad=requires_grad, name=name)


@dataclass
class Parameter:
    data: Tensor
    requires_grad: bool = True

    def __post_init__(self):
        self.data.requires_grad = self.requires_grad

    @property
    def grad(self):
        return self.data.grad

    def zero_grad(self):
        self.data.grad = None


class Module:
    def parameters(self):
        for _, value in self.__dict__.items():
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        yield from item.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()


class Linear(Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        self.weight = Parameter(create_forge_random_tensor((input_size, output_size), -0.1, 0.1, True, name="weight"))
        self.bias = Parameter(create_forge_zero_tensor((output_size,), True, name="bias")) if bias else None

    def __call__(self, inputs):
        tensor = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        output = tensor.matmul(self.weight.data)
        if self.bias is not None:
            output = output + self.bias.data
        return output


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output

    def parameters(self):
        for layer in self.layers:
            if isinstance(layer, Module):
                yield from layer.parameters()


class ReLU(Module):
    def __call__(self, inputs):
        tensor = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        return tensor.relu()


class Sigmoid(Module):
    def __call__(self, inputs):
        tensor = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        return tensor.sigmoid()


class Tanh(Module):
    def __call__(self, inputs):
        tensor = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        return tensor.tanh()


class Softmax(Module):
    def __init__(self, axis: int = -1):
        self.axis = axis

    def __call__(self, inputs):
        tensor = inputs if isinstance(inputs, Tensor) else Tensor(inputs)
        return tensor.softmax(axis=self.axis)


class SGD:
    def __init__(self, parameters: Iterable[Parameter], learning_rate: float = 0.01, momentum: float = 0.0):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {id(parameter): 0.0 for parameter in self.parameters}

    def step(self):
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            grad = parameter.grad if isinstance(parameter.grad, (int, float)) else sum(_flatten(parameter.grad))
            velocity = self.momentum * self.velocity[id(parameter)] + grad
            self.velocity[id(parameter)] = velocity
            parameter.data = parameter.data - self.learning_rate * velocity

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()


class Adam:
    def __init__(self, parameters: Iterable[Parameter], learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step_count = 0
        self.first_moment = {id(parameter): 0.0 for parameter in self.parameters}
        self.second_moment = {id(parameter): 0.0 for parameter in self.parameters}

    def step(self):
        self.step_count += 1
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            grad = parameter.grad if isinstance(parameter.grad, (int, float)) else sum(_flatten(parameter.grad))
            pid = id(parameter)
            self.first_moment[pid] = self.beta1 * self.first_moment[pid] + (1 - self.beta1) * grad
            self.second_moment[pid] = self.beta2 * self.second_moment[pid] + (1 - self.beta2) * (grad * grad)
            corrected_first = self.first_moment[pid] / (1 - self.beta1 ** self.step_count)
            corrected_second = self.second_moment[pid] / (1 - self.beta2 ** self.step_count)
            update = self.learning_rate * corrected_first / (math.sqrt(corrected_second) + self.epsilon)
            parameter.data = parameter.data - update

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()


def create_linear_layer(input_size: int, output_size: int, bias: bool = True):
    return Linear(input_size, output_size, bias=bias)


def create_sequential_model(*layers):
    return Sequential(*layers)


def create_relu_layer():
    return ReLU()


def create_sigmoid_layer():
    return Sigmoid()


def create_tanh_layer():
    return Tanh()


def create_softmax_layer(axis: int = -1):
    return Softmax(axis=axis)


def create_sgd_optimizer(parameters, learning_rate: float = 0.01, momentum: float = 0.0):
    return SGD(parameters, learning_rate=learning_rate, momentum=momentum)


def create_adam_optimizer(parameters, learning_rate: float = 0.001):
    return Adam(parameters, learning_rate=learning_rate)


def create_mean_squared_error_loss(predicted, target):
    predicted_tensor = predicted if isinstance(predicted, Tensor) else Tensor(predicted)
    target_tensor = target if isinstance(target, Tensor) else Tensor(target)
    diff = predicted_tensor - target_tensor
    return (diff * diff).mean()


def create_cross_entropy_loss(logits, target_index: int):
    logits_tensor = logits if isinstance(logits, Tensor) else Tensor(logits)
    probabilities = logits_tensor.softmax()
    target = [0.0 for _ in probabilities.to_list()]
    target[target_index] = 1.0
    loss = 0.0
    for predicted_value, target_value in zip(probabilities.to_list(), target):
        loss += -target_value * math.log(max(predicted_value, 1e-8))
    return Tensor(loss, name="cross_entropy_loss")


def _flatten(values):
    if isinstance(values, Tensor):
        return _flatten(values.to_list())
    if isinstance(values, (list, tuple)):
        out = []
        for item in values:
            out.extend(_flatten(item))
        return out
    return [float(values)]
