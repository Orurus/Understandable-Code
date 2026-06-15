"""Tensor and matrix helpers for UnderstandableCode ML experiments.

This module keeps the API intentionally verbose, but the implementation is now
feature-rich enough to support a small PyTorch-like workflow:
- tensor creation and inspection
- broadcasting-aware elementwise math
- indexing, reshape, transpose, flatten, squeeze, unsqueeze
- reductions such as sum, mean, max, min
- a lightweight reverse-mode autograd engine
- matrix multiplication and activation helpers
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union


Number = Union[int, float]
NestedTensor = Union[Number, List["NestedTensor"]]


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _to_nested_list(value: Any) -> Any:
    if isinstance(value, Tensor):
        return deepcopy(value.data)
    if isinstance(value, tuple):
        return [_to_nested_list(item) for item in value]
    if isinstance(value, list):
        return [_to_nested_list(item) for item in value]
    return value


def _ensure_numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    return float(value)


def _infer_shape(data: Any) -> List[int]:
    if isinstance(data, Tensor):
        return list(data.shape)
    if not _is_sequence(data):
        return []
    data = list(data)
    if not data:
        return [0]
    inner = _infer_shape(data[0])
    return [len(data)] + inner


def _flatten(data: Any) -> List[float]:
    if isinstance(data, Tensor):
        return _flatten(data.data)
    if _is_sequence(data):
        flat: List[float] = []
        for item in data:
            flat.extend(_flatten(item))
        return flat
    return [_ensure_numeric(data)]


def _reshape_from_flat(flat_values: Sequence[float], shape: Sequence[int]) -> Any:
    if not shape:
        return flat_values[0] if flat_values else 0.0
    if len(shape) == 1:
        return [flat_values[index] for index in range(shape[0])]
    step = 1
    for size in shape[1:]:
        step *= size
    return [
        _reshape_from_flat(flat_values[index * step:(index + 1) * step], shape[1:])
        for index in range(shape[0])
    ]


def _numel(shape: Sequence[int]) -> int:
    total = 1
    for size in shape:
        total *= size
    return total


def _broadcast_shape(left: Sequence[int], right: Sequence[int]) -> List[int]:
    result: List[int] = []
    left_rev = list(reversed(left))
    right_rev = list(reversed(right))
    for index in range(max(len(left_rev), len(right_rev))):
        l_value = left_rev[index] if index < len(left_rev) else 1
        r_value = right_rev[index] if index < len(right_rev) else 1
        if l_value == r_value or l_value == 1 or r_value == 1:
            result.append(max(l_value, r_value))
        else:
            raise ValueError(f"Incompatible shapes for broadcasting: {list(left)} and {list(right)}")
    return list(reversed(result))


def _broadcast_to_shape(data: Any, source_shape: Sequence[int], target_shape: Sequence[int]) -> Any:
    if list(source_shape) == list(target_shape):
        return deepcopy(_to_nested_list(data))
    if not target_shape:
        return _flatten(data)[0]
    if not source_shape:
        return _reshape_from_flat([_flatten(data)[0]], target_shape)
    if len(source_shape) < len(target_shape):
        padding = [1] * (len(target_shape) - len(source_shape))
        source_shape = padding + list(source_shape)
        data = _reshape_from_flat(_flatten(data), source_shape)
    if len(target_shape) == 1:
        if source_shape[0] == 1:
            value = _flatten(data)[0]
            return [value for _ in range(target_shape[0])]
        return list(_to_nested_list(data))
    if source_shape[0] == target_shape[0]:
        return [
            _broadcast_to_shape(item, source_shape[1:], target_shape[1:])
            for item in list(data)
        ]
    if source_shape[0] == 1:
        item = list(data)[0]
        return [
            _broadcast_to_shape(item, source_shape[1:], target_shape[1:])
            for _ in range(target_shape[0])
        ]
    raise ValueError(f"Cannot broadcast shape {list(source_shape)} to {list(target_shape)}")


def _elementwise_apply(left: Any, right: Any, fn: Callable[[float, float], float]) -> Any:
    left_is_seq = _is_sequence(left)
    right_is_seq = _is_sequence(right)
    if not left_is_seq and not right_is_seq:
        return fn(_ensure_numeric(left), _ensure_numeric(right))
    left_list = list(left) if left_is_seq else [left]
    right_list = list(right) if right_is_seq else [right]
    if len(left_list) == len(right_list):
        return [_elementwise_apply(l_item, r_item, fn) for l_item, r_item in zip(left_list, right_list)]
    if len(left_list) == 1:
        return [_elementwise_apply(left_list[0], r_item, fn) for r_item in right_list]
    if len(right_list) == 1:
        return [_elementwise_apply(l_item, right_list[0], fn) for l_item in left_list]
    raise ValueError("Unable to apply elementwise operation to mismatched shapes")


def _elementwise_unary(data: Any, fn: Callable[[float], float]) -> Any:
    if _is_sequence(data):
        return [_elementwise_unary(item, fn) for item in data]
    return fn(_ensure_numeric(data))


def _transpose_2d(data: Sequence[Sequence[float]]) -> List[List[float]]:
    if not data:
        return []
    return [list(column) for column in zip(*data)]


def _zeros_like(data: Any) -> Any:
    if _is_sequence(data):
        return [_zeros_like(item) for item in data]
    return 0.0


def _add_in_place(target: Any, source: Any) -> Any:
    if _is_sequence(target) and _is_sequence(source):
        for index, item in enumerate(source):
            target[index] = _add_in_place(target[index], item)
        return target
    return _ensure_numeric(target) + _ensure_numeric(source)


def _reduce_grad_to_shape(gradient: Any, target_shape: Sequence[int]) -> Any:
    gradient_shape = _infer_shape(gradient)
    if list(gradient_shape) == list(target_shape):
        return deepcopy(gradient)
    if not target_shape:
        return sum(_flatten(gradient))
    if len(target_shape) < len(gradient_shape):
        for _ in range(len(gradient_shape) - len(target_shape)):
            target_shape = [1] + list(target_shape)
    result = deepcopy(gradient)
    while _infer_shape(result) != list(target_shape):
        result_shape = _infer_shape(result)
        if len(result_shape) > len(target_shape):
            result = [sum(_flatten(result))]
            continue
        break
    return _reshape_from_flat(_flatten(result), target_shape)


def compute_tensor_shape(tensor) -> List[int]:
    """Infer the shape of a nested list tensor."""
    return _infer_shape(tensor)


def validate_tensor_shapes_for_broadcasting(left_shape: Sequence[int], right_shape: Sequence[int]) -> bool:
    """Check whether two shapes can participate in broadcasting."""
    try:
        _broadcast_shape(left_shape, right_shape)
    except ValueError:
        return False
    return True


def apply_tensor_broadcast_addition(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> List[List[float]]:
    """Add a vector across each row of a matrix with shape validation."""
    matrix_shape = compute_tensor_shape(matrix)
    vector_shape = compute_tensor_shape(vector)
    if not validate_tensor_shapes_for_broadcasting(matrix_shape, vector_shape):
        raise ValueError("Incompatible shapes for broadcasting")
    target_shape = _broadcast_shape(matrix_shape, vector_shape)
    matrix_broadcast = _broadcast_to_shape(matrix, matrix_shape, target_shape)
    vector_broadcast = _broadcast_to_shape(vector, vector_shape, target_shape)
    return _elementwise_apply(matrix_broadcast, vector_broadcast, lambda a, b: a + b)


def compute_tensor_mean_value(tensor: Iterable) -> float:
    """Compute the mean value of all elements stored inside a tensor."""
    flat_values = _flatten(tensor)
    return sum(flat_values) / len(flat_values) if flat_values else 0.0


def new_matrix_with_zeros(rows: int, columns: int) -> List[List[int]]:
    """Create a new matrix filled with zeros."""
    return [[0 for _ in range(columns)] for _ in range(rows)]


def new_matrix_with_ones(rows: int, columns: int) -> List[List[int]]:
    """Create a new matrix filled with ones."""
    return [[1 for _ in range(columns)] for _ in range(rows)]


def new_matrix_with_identity(size: int) -> List[List[int]]:
    """Create a new identity matrix with ones on the diagonal."""
    matrix = new_matrix_with_zeros(size, size)
    for index in range(size):
        matrix[index][index] = 1
    return matrix


def new_matrix_with_random_values(rows: int, columns: int, minimum: float = 0.0, maximum: float = 1.0) -> List[List[float]]:
    """Create a new matrix filled with random floating-point values."""
    return [[random.uniform(minimum, maximum) for _ in range(columns)] for _ in range(rows)]


def new_tensor_with_zeros(shape: Sequence[int]) -> List:
    """Create a new tensor filled with zeros using a shape tuple."""
    if not shape:
        return 0
    if len(shape) == 1:
        return [0 for _ in range(shape[0])]
    return [new_tensor_with_zeros(shape[1:]) for _ in range(shape[0])]


def new_tensor_with_ones(shape: Sequence[int]) -> List:
    """Create a new tensor filled with ones using a shape tuple."""
    if not shape:
        return 1
    if len(shape) == 1:
        return [1 for _ in range(shape[0])]
    return [new_tensor_with_ones(shape[1:]) for _ in range(shape[0])]


def new_tensor_with_random_values(shape: Sequence[int], minimum: float = 0.0, maximum: float = 1.0) -> List:
    """Create a new tensor filled with random floating-point values."""
    if not shape:
        return random.uniform(minimum, maximum)
    if len(shape) == 1:
        return [random.uniform(minimum, maximum) for _ in range(shape[0])]
    return [new_tensor_with_random_values(shape[1:], minimum, maximum) for _ in range(shape[0])]


def add_matrix_to_matrix(left_matrix: Sequence[Sequence[float]], right_matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    """Add one matrix to another matrix element by element."""
    return [[left + right for left, right in zip(row_left, row_right)]
            for row_left, row_right in zip(left_matrix, right_matrix)]


def multiply_matrix_by_matrix(left_matrix: Sequence[Sequence[float]], right_matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    """Multiply two matrices using standard matrix multiplication."""
    rows = len(left_matrix)
    cols = len(right_matrix[0]) if right_matrix else 0
    result = new_matrix_with_zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            total = 0.0
            for k in range(len(right_matrix)):
                total += left_matrix[i][k] * right_matrix[k][j]
            result[i][j] = total
    return result


@dataclass
class _AutogradContext:
    parents: Tuple["Tensor", ...]
    op: str
    backward: Callable[[Any], Tuple[Any, ...]]


class Tensor:
    """A small tensor object with shape tracking, broadcasting, and autograd."""

    def __init__(
        self,
        data,
        name: str = "tensor",
        grad=None,
        requires_grad: bool = False,
        dtype: type = float,
    ):
        self.data = _to_nested_list(data)
        self.shape = compute_tensor_shape(self.data)
        self.name = name
        self.grad = deepcopy(grad)
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.history: List[str] = []
        self._ctx: Optional[_AutogradContext] = None

    def __repr__(self) -> str:
        return f"Tensor(name={self.name!r}, shape={self.shape!r}, data={self.data!r})"

    def to_list(self):
        return deepcopy(self.data)

    def clone(self, name: Optional[str] = None):
        return Tensor(self.data, name=name or self.name, grad=self.grad, requires_grad=self.requires_grad, dtype=self.dtype)

    def detach(self):
        return Tensor(self.data, name=self.name, grad=None, requires_grad=False, dtype=self.dtype)

    def item(self) -> float:
        flat = _flatten(self.data)
        if len(flat) != 1:
            raise ValueError("item() is only valid for scalar tensors")
        return flat[0]

    def flatten(self):
        return Tensor(_flatten(self.data), name=f"{self.name}.flatten", requires_grad=self.requires_grad, dtype=self.dtype)

    def reshape(self, *shape: int):
        if len(shape) == 1 and _is_sequence(shape[0]):
            shape = tuple(shape[0])  # type: ignore[assignment]
        flat = _flatten(self.data)
        if _numel(shape) != len(flat):
            raise ValueError("Cannot reshape tensor to incompatible shape")
        return Tensor(_reshape_from_flat(flat, shape), name=f"{self.name}.reshape", requires_grad=self.requires_grad, dtype=self.dtype)

    def transpose(self):
        if len(self.shape) != 2:
            raise ValueError("transpose() currently supports only 2D tensors")
        return Tensor(_transpose_2d(self.data), name=f"{self.name}.transpose", requires_grad=self.requires_grad, dtype=self.dtype)

    T = property(transpose)

    def unsqueeze(self, dim: int = 0):
        def insert_dimension(values, remaining_dim: int):
            if remaining_dim == 0:
                return [values]
            if not _is_sequence(values):
                return [values] if remaining_dim == 1 else insert_dimension([values], remaining_dim - 1)
            return [insert_dimension(item, remaining_dim - 1) for item in values]

        if dim < 0:
            dim += len(self.shape) + 1
        return Tensor(insert_dimension(self.to_list(), dim), name=f"{self.name}.unsqueeze", requires_grad=self.requires_grad, dtype=self.dtype)

    def squeeze(self):
        data = self.to_list()
        while _is_sequence(data) and len(data) == 1:
            data = data[0]
        return Tensor(data, name=f"{self.name}.squeeze", requires_grad=self.requires_grad, dtype=self.dtype)

    def sum(self, axis: Optional[int] = None):
        if axis is None:
            value = sum(_flatten(self.data))
            return Tensor(value, name=f"{self.name}.sum", requires_grad=self.requires_grad, dtype=self.dtype)
        if len(self.shape) == 1:
            return Tensor(sum(_flatten(self.data)), name=f"{self.name}.sum", requires_grad=self.requires_grad, dtype=self.dtype)
        if axis == 0:
            return Tensor([sum(column) for column in zip(*self.data)], name=f"{self.name}.sum", requires_grad=self.requires_grad, dtype=self.dtype)
        if axis == 1:
            return Tensor([sum(row) for row in self.data], name=f"{self.name}.sum", requires_grad=self.requires_grad, dtype=self.dtype)
        raise ValueError("axis out of range")

    def mean(self, axis: Optional[int] = None):
        if axis is None:
            flat = _flatten(self.data)
            return Tensor(sum(flat) / len(flat) if flat else 0.0, name=f"{self.name}.mean", requires_grad=self.requires_grad, dtype=self.dtype)
        summed = self.sum(axis=axis)
        divisor = self.shape[axis]
        return summed / divisor

    def relu(self):
        return Tensor(_elementwise_unary(self.data, lambda value: 0.0 if value < 0.0 else value), name=f"{self.name}.relu", requires_grad=self.requires_grad, dtype=self.dtype)

    def sigmoid(self):
        return Tensor(_elementwise_unary(self.data, lambda value: 1.0 / (1.0 + math.exp(-value))), name=f"{self.name}.sigmoid", requires_grad=self.requires_grad, dtype=self.dtype)

    def tanh(self):
        return Tensor(_elementwise_unary(self.data, math.tanh), name=f"{self.name}.tanh", requires_grad=self.requires_grad, dtype=self.dtype)

    def softmax(self, axis: int = -1):
        if not self.shape:
            return Tensor([1.0], name=f"{self.name}.softmax", requires_grad=self.requires_grad, dtype=self.dtype)
        if len(self.shape) == 1 or axis in (-1, 0):
            flat = _flatten(self.data)
            max_value = max(flat)
            exponentials = [math.exp(value - max_value) for value in flat]
            total = sum(exponentials) or 1.0
            return Tensor([value / total for value in exponentials], name=f"{self.name}.softmax", requires_grad=self.requires_grad, dtype=self.dtype)
        if len(self.shape) == 2 and axis == 1:
            rows = []
            for row in self.data:
                max_value = max(row)
                exponentials = [math.exp(value - max_value) for value in row]
                total = sum(exponentials) or 1.0
                rows.append([value / total for value in exponentials])
            return Tensor(rows, name=f"{self.name}.softmax", requires_grad=self.requires_grad, dtype=self.dtype)
        raise ValueError("softmax() currently supports 1D tensors or axis=1 for 2D tensors")

    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("matmul() currently supports 2D tensors only")
        result = multiply_matrix_by_matrix(self.data, other.data)
        out = Tensor(result, name=f"{self.name}.matmul", requires_grad=self.requires_grad or other.requires_grad, dtype=self.dtype)

        def backward(grad_output):
            left_grad = multiply_matrix_by_matrix(grad_output, _transpose_2d(other.data))
            right_grad = multiply_matrix_by_matrix(_transpose_2d(self.data), grad_output)
            return left_grad, right_grad

        if out.requires_grad:
            out._ctx = _AutogradContext(parents=(self, other), op="matmul", backward=backward)
        return out

    def backward(self, gradient=None):
        if not self.requires_grad:
            return
        if gradient is None:
            gradient = 1.0 if not self.shape else _ones_like(self.data)
        self._accumulate_grad(gradient)
        visited = set()
        stack = [self]
        while stack:
            tensor = stack.pop()
            if tensor._ctx is None:
                continue
            ctx = tensor._ctx
            parent_grads = ctx.backward(tensor.grad)
            for parent, parent_grad in zip(ctx.parents, parent_grads):
                if not parent.requires_grad:
                    continue
                parent._accumulate_grad(parent_grad)
                if parent not in visited:
                    visited.add(parent)
                    stack.append(parent)

    def _accumulate_grad(self, gradient):
        if self.grad is None:
            self.grad = deepcopy(gradient)
        else:
            self.grad = _add_in_place(deepcopy(self.grad), gradient)

    def _binary_op(self, other, name: str, fn: Callable[[float, float], float], backward_fn: Callable[[Any, Any], Tuple[Any, Any]]):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out_shape = _broadcast_shape(self.shape, other.shape)
        left = _broadcast_to_shape(self.data, self.shape, out_shape)
        right = _broadcast_to_shape(other.data, other.shape, out_shape)
        result = _elementwise_apply(left, right, fn)
        out = Tensor(result, name=f"{self.name}.{name}", requires_grad=self.requires_grad or other.requires_grad, dtype=self.dtype)

        if out.requires_grad:
            def backward(grad_output):
                left_grad, right_grad = backward_fn(grad_output, left, right)
                return (
                    _reduce_grad_to_shape(left_grad, self.shape),
                    _reduce_grad_to_shape(right_grad, other.shape),
                )

            out._ctx = _AutogradContext(parents=(self, other), op=name, backward=backward)
        return out

    def __add__(self, other):
        return self._binary_op(other, "add", lambda a, b: a + b, lambda grad, _left, _right: (grad, grad))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, "sub", lambda a, b: a - b, lambda grad, _left, _right: (grad, _elementwise_unary(grad, lambda value: -value)))

    def __rsub__(self, other):
        return Tensor(other).__sub__(self)

    def __mul__(self, other):
        return self._binary_op(other, "mul", lambda a, b: a * b, lambda grad, left, right: (_elementwise_apply(grad, right, lambda g, r: g * r), _elementwise_apply(grad, left, lambda g, l: g * l)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_op(other, "div", lambda a, b: a / b, lambda grad, left, right: (_elementwise_apply(grad, right, lambda g, r: g / r), _elementwise_apply(grad, left, lambda g, l: -g * l / (l * l if l else 1.0))))

    def __neg__(self):
        return Tensor(_elementwise_unary(self.data, lambda value: -value), name=f"{self.name}.neg", requires_grad=self.requires_grad, dtype=self.dtype)

    def __getitem__(self, item):
        return Tensor(self.data[item], name=f"{self.name}.slice", requires_grad=self.requires_grad, dtype=self.dtype)

    def astype(self, dtype: type):
        return Tensor(_elementwise_unary(self.data, dtype), name=f"{self.name}.astype", requires_grad=self.requires_grad, dtype=dtype)


def _ones_like(data: Any) -> Any:
    if _is_sequence(data):
        return [_ones_like(item) for item in data]
    return 1.0


def create_neural_network_layer_with_random_weights(rows: int, columns: int):
    """Create a new neural network layer matrix filled with random weights."""
    return new_matrix_with_random_values(rows, columns, -0.1, 0.1)


def create_neural_network_layer_with_zeros(rows: int, columns: int):
    """Create a new neural network layer matrix filled with zeros."""
    return new_matrix_with_zeros(rows, columns)


def create_neural_network_layer_with_ones(rows: int, columns: int):
    """Create a new neural network layer matrix filled with ones."""
    return new_matrix_with_ones(rows, columns)


def apply_relu_activation_function(matrix):
    """Apply the ReLU activation function element by element."""
    return [[0.0 if value < 0.0 else value for value in row] for row in matrix]


def apply_sigmoid_activation_function(matrix):
    """Apply the sigmoid activation function element by element."""
    result = []
    for row in matrix:
        result_row = []
        for value in row:
            exponent = math.exp(-value)
            result_row.append(1.0 / (1.0 + exponent))
        result.append(result_row)
    return result


def apply_softmax_activation_function(vector):
    """Apply the softmax activation function to a vector."""
    max_value = max(vector)
    exponentials = [math.exp(value - max_value) for value in vector]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]


def create_attention_score_matrix(query_matrix, key_matrix):
    """Create a simple attention score matrix from query and key matrices."""
    return multiply_matrix_by_matrix(query_matrix, key_matrix)


def create_transformer_position_encoding(sequence_length: int, embedding_size: int):
    """Create a simple sinusoidal positional encoding matrix for transformers."""
    encoding = new_matrix_with_zeros(sequence_length, embedding_size)
    for position in range(sequence_length):
        for dimension in range(embedding_size):
            if dimension % 2 == 0:
                encoding[position][dimension] = math.sin(position / (10000 ** (dimension / embedding_size)))
            else:
                encoding[position][dimension] = math.cos(position / (10000 ** ((dimension - 1) / embedding_size)))
    return encoding


def create_simple_snn_neuron_output(input_values):
    """Create a simple spiking neural network neuron output from input values."""
    return [1.0 if value > 0.0 else 0.0 for value in input_values]


def create_simple_transformer_feed_forward_layer(input_vector):
    """Create a simple feed-forward transform for transformer-style layers."""
    return [value * 2.0 for value in input_vector]


def create_simple_language_model_token_embedding(vocabulary_size: int, embedding_size: int):
    """Create a simple token embedding matrix for language model experiments."""
    return new_matrix_with_random_values(vocabulary_size, embedding_size, -0.05, 0.05)


def create_neural_network_bias_vector(size: int):
    """Create a new bias vector filled with zeros for a neural network layer."""
    return [0.0 for _ in range(size)]


def apply_tanh_activation_function(matrix):
    """Apply the hyperbolic tangent activation function element by element."""
    result = []
    for row in matrix:
        result.append([math.tanh(value) for value in row])
    return result


def apply_leaky_relu_activation_function(matrix, negative_slope: float = 0.01):
    """Apply the leaky ReLU activation function element by element."""
    return [[value if value >= 0.0 else negative_slope * value for value in row] for row in matrix]


def compute_mean_squared_error(predicted_matrix, target_matrix):
    """Compute the mean squared error between predicted values and target values."""
    total = 0.0
    count = 0
    for row_pred, row_target in zip(predicted_matrix, target_matrix):
        for value_pred, value_target in zip(row_pred, row_target):
            total += (value_pred - value_target) ** 2
            count += 1
    return total / count if count else 0.0


def compute_binary_cross_entropy_loss(predicted_matrix, target_matrix):
    """Compute the binary cross entropy loss for probability-style predictions."""
    total = 0.0
    count = 0
    for row_pred, row_target in zip(predicted_matrix, target_matrix):
        for value_pred, value_target in zip(row_pred, row_target):
            eps = 1e-7
            clipped = min(max(value_pred, eps), 1.0 - eps)
            total += -(value_target * math.log(clipped) + (1.0 - value_target) * math.log(1.0 - clipped))
            count += 1
    return total / count if count else 0.0


def apply_layer_normalization_vector(vector):
    """Apply a simple layer normalization step to a vector of values."""
    mean_value = sum(vector) / len(vector) if vector else 0.0
    variance = sum((value - mean_value) ** 2 for value in vector) / len(vector) if vector else 0.0
    std_value = math.sqrt(variance)
    return [(value - mean_value) / (std_value + 1e-6) for value in vector]


def create_dropout_mask_vector(vector, dropout_rate: float = 0.1):
    """Create a binary dropout mask for a vector of values."""
    return [0.0 if random.random() < dropout_rate else 1.0 for _ in vector]


def create_residual_connection_matrix(primary_matrix, residual_matrix):
    """Create a residual connection by adding a primary pathway to its residual pathway."""
    return [[left + right for left, right in zip(left_row, right_row)]
            for left_row, right_row in zip(primary_matrix, residual_matrix)]


def create_scaled_dot_product_attention_matrix(query_matrix, key_matrix, value_matrix):
    """Create row-wise scaled dot-product attention using real query/key/value semantics."""
    query_shape = compute_tensor_shape(query_matrix)
    key_shape = compute_tensor_shape(key_matrix)
    value_shape = compute_tensor_shape(value_matrix)

    if len(query_shape) != 2 or len(key_shape) != 2 or len(value_shape) != 2:
        raise ValueError("Attention inputs must be 2D matrices")
    if query_shape[1] != key_shape[0]:
        raise ValueError("Query and key dimensions are incompatible")
    if key_shape[0] != value_shape[0]:
        raise ValueError("Key and value sequence lengths must match")

    scores = []
    for query_row in query_matrix:
        row_scores = []
        for key_row in key_matrix:
            score = sum(q * k for q, k in zip(query_row, key_row))
            row_scores.append(score)
        scores.append(row_scores)

    scale_factor = 1.0 / math.sqrt(len(query_matrix[0]) if query_matrix else 1)
    scaled_scores = [[value * scale_factor for value in row] for row in scores]

    attention_rows = []
    for row in scaled_scores:
        weights = apply_softmax_activation_function(row)
        weighted_values = [sum(weight * value for weight, value in zip(weights, column_values))
                           for column_values in zip(*value_matrix)]
        attention_rows.append(weighted_values)

    return attention_rows


def create_simple_transformer_block_output(input_matrix):
    """Create a simple transformer-style block output from an input matrix."""
    return [[value * 1.1 for value in row] for row in input_matrix]


def create_spiking_neuron_threshold_output(input_values, threshold: float = 0.5):
    """Create a spiking neural network output by thresholding input values."""
    return [1.0 if value >= threshold else 0.0 for value in input_values]


def create_simple_language_model_context_window(token_ids, window_size: int):
    """Create a simple context window slice for language model experiments."""
    return token_ids[:window_size]


def create_weight_update_using_error_signal(weights, targets, learning_rate: float = 0.01):
    """Create a simple weight update using an error signal and learning rate."""
    return [[weight - learning_rate * (weight - target) for weight, target in zip(row_weights, row_targets)]
            for row_weights, row_targets in zip(weights, targets)]


def apply_weight_decay_regularization(weights, decay_rate: float = 0.01):
    """Apply weight decay regularization to reduce large parameter values."""
    return [[value * (1.0 - decay_rate) for value in row] for row in weights]


def create_batch_of_input_vectors(batch_input_vectors):
    """Create a batch of input vectors for mini-batch style processing."""
    return [list(row) for row in batch_input_vectors]


def apply_value_clipping_to_matrix(matrix, lower_bound: float = -10.0, upper_bound: float = 10.0):
    """Clip all values in a matrix to safe numeric bounds."""
    return [[max(lower_bound, min(upper_bound, value)) for value in row] for row in matrix]


def apply_safe_division_operation(numerator, denominator, epsilon: float = 1e-8):
    """Perform safe division by guarding against zero denominators."""
    return numerator / (denominator + epsilon)


def create_hebbian_learning_update(matrix, activation_vector):
    """Create a simple Hebbian learning update using a correlation-style rule."""
    return [[value * activation for activation in activation_vector] for value in matrix]


def create_gradient_descent_optimization_step(weights, gradients, learning_rate: float = 0.01):
    """Create a simple gradient descent optimization step for parameter updates."""
    return [[weight - learning_rate * gradient for weight, gradient in zip(row_weights, row_gradients)]
            for row_weights, row_gradients in zip(weights, gradients)]


def create_momentum_optimization_step(weights, gradients, velocity, learning_rate: float = 0.01, momentum_rate: float = 0.9):
    """Create a simple momentum-based optimization step for smoother training."""
    updated_velocity = [[momentum_rate * vel + learning_rate * grad for vel, grad in zip(row_velocity, row_gradients)]
                        for row_velocity, row_gradients in zip(velocity, gradients)]
    updated_weights = [[weight - step for weight, step in zip(row_weights, row_velocity)]
                       for row_weights, row_velocity in zip(weights, updated_velocity)]
    return updated_weights, updated_velocity


def create_simple_model_state_dictionary(model_name: str, parameter_count: int):
    """Create a simple dictionary-like model state object for a named model."""
    return {"model_name": model_name, "parameter_count": parameter_count, "ready": True}


def create_tensor_state_object_with_shape_and_values(shape, values):
    """Create a persistent tensor state object that tracks shape and current values."""
    return {"shape": list(shape), "values": [list(row) for row in values], "persistent": True}


def create_timestep_state_transition(current_state, delta_state, timestep_factor: float = 1.0):
    """Create a simple timestep-based state transition using current state and deltas."""
    return [[value + timestep_factor * delta for value, delta in zip(row_values, row_deltas)]
            for row_values, row_deltas in zip(current_state, delta_state)]


def apply_simple_broadcasting_rule_to_matrix(matrix, vector):
    """Apply a simple broadcasting rule by adding a vector across each row of a matrix."""
    return apply_tensor_broadcast_addition(matrix, vector)


def create_computation_graph_node(operation_name: str, input_names):
    """Create a simple computation graph node that records an operation and its inputs."""
    return {"operation": operation_name, "inputs": list(input_names), "output": None}


def create_simple_autograd_gradient_trace(operation_name: str, gradient_value: float):
    """Create a simple autograd-style gradient trace record for a computation step."""
    return {"operation": operation_name, "gradient": gradient_value}


def create_autograd_graph_node(operation_name: str, input_names, gradient: float = None):
    """Create an autograd-style graph node with dependency metadata and gradient state."""
    return {"operation": operation_name, "inputs": list(input_names), "gradient": gradient, "backward": False}


def record_tensor_operation_in_graph(operation_name: str, inputs, output_tensor):
    """Record a simple graph edge for tensor operations so the graph becomes automatic."""
    return create_autograd_graph_node(operation_name, list(inputs), gradient=getattr(output_tensor, "grad", None))


def create_reusable_model_module_container(model_name: str, layers):
    """Create a reusable model module container that groups model layers under one name."""
    return {"model_name": model_name, "layers": list(layers), "composition": True}


def create_optimizer_state_object(optimizer_name: str, parameters, learning_rate: float = 0.01):
    """Create a stateful optimizer object for training loops."""
    return {
        "optimizer": optimizer_name,
        "parameters": dict(parameters),
        "learning_rate": learning_rate,
        "velocity": {},
        "step": 0,
    }


def apply_optimizer_step_with_state(weights, gradients, optimizer_state):
    """Apply a simple SGD update and return the new weights plus updated optimizer state."""
    updated_weights = [[weight - optimizer_state["learning_rate"] * gradient
                        for weight, gradient in zip(row_weights, row_gradients)]
                       for row_weights, row_gradients in zip(weights, gradients)]
    optimizer_state = dict(optimizer_state)
    optimizer_state["step"] = optimizer_state.get("step", 0) + 1
    optimizer_state["parameters"] = {"w": updated_weights}
    optimizer_state["history"] = optimizer_state.get("history", []) + [{"step": optimizer_state["step"], "weights": updated_weights}]
    return updated_weights, optimizer_state


def create_training_loop_state(model_name: str, optimizer_name: str, parameters, learning_rate: float = 0.01):
    """Create a simple training-loop state object with optimizer metadata and history."""
    return {
        "model_name": model_name,
        "optimizer_name": optimizer_name,
        "optimizer_state": create_optimizer_state_object(optimizer_name, parameters, learning_rate),
        "history": [],
    }


def run_simple_training_loop_with_optimizer(weights, gradients, training_state):
    """Run one simple training step through the optimizer abstraction and record the history."""
    updated_weights, optimizer_state = apply_optimizer_step_with_state(weights, gradients, training_state["optimizer_state"])
    training_state = dict(training_state)
    training_state["optimizer_state"] = optimizer_state
    training_state["history"] = training_state.get("history", []) + [{"step": optimizer_state["step"], "weights": updated_weights}]
    training_state["weights"] = updated_weights
    return training_state
