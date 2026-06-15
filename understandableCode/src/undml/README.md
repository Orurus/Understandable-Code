# undml

`undml` is the machine-learning side of UnderstandableCode. It keeps the code split into
small, readable pieces so the language runtime can import tensor helpers, higher-level ML
helpers, and local model-building tools without needing a large external framework.

## What lives here

- `load`: module loading for `.und` files and Python helpers.
- `tensor`: core tensor and matrix operations, including a lightweight `Tensor` class.
- `adv`: higher-level neural network, attention, optimizer, and training helpers.
- `tensorforge`: local tensor, layer, module, and optimizer helpers with a framework-style API.

## Most used pieces

If you only learn a few parts first, start with these:

### `undml.tensor`

This is the foundation layer. It provides:

- tensor and matrix creation
- shape inference
- broadcasting checks
- elementwise math helpers
- matrix multiplication
- a lightweight autograd-capable `Tensor`

Useful entry points:

- `Tensor(data, requires_grad=False, name="tensor")`
- `compute_tensor_shape(tensor)`
- `validate_tensor_shapes_for_broadcasting(left_shape, right_shape)`
- `apply_tensor_broadcast_addition(matrix, vector)`
- `new_matrix_with_zeros(rows, columns)`
- `new_matrix_with_ones(rows, columns)`
- `new_matrix_with_identity(size)`
- `new_tensor_with_zeros(shape)`
- `new_tensor_with_ones(shape)`
- `new_tensor_with_random_values(shape, minimum, maximum)`

The `Tensor` object supports:

- `to_list()`
- `clone()`
- `detach()`
- `item()`
- `flatten()`
- `reshape(...)`
- `transpose()` and `T`
- `unsqueeze(dim)`
- `squeeze()`
- `sum(axis=None)`
- `mean(axis=None)`
- `relu()`
- `sigmoid()`
- `tanh()`
- `softmax(axis=-1)`
- `matmul(other)`
- `backward()`

That makes it the most important layer for experiments that need tensor-like behavior.

### `undml.adv`

This is the “bigger ML helper” layer. It builds on the tensor utilities and provides:

- activation helpers
- attention helpers
- positional encodings
- residual connections
- loss functions
- optimizer state helpers
- batch and clipping helpers
- simple graph and training-state helpers

Most useful functions:

- `apply_relu_activation_function(matrix)`
- `apply_sigmoid_activation_function(matrix)`
- `apply_tanh_activation_function(matrix)`
- `apply_softmax_activation_function(vector)`
- `create_scaled_dot_product_attention_matrix(query_matrix, key_matrix, value_matrix)`
- `create_transformer_position_encoding(sequence_length, embedding_size)`
- `compute_mean_squared_error(predicted_matrix, target_matrix)`
- `compute_binary_cross_entropy_loss(predicted_matrix, target_matrix)`
- `create_optimizer_state_object(optimizer_name, parameters, learning_rate)`
- `apply_optimizer_step_with_state(weights, gradients, optimizer_state)`
- `create_training_loop_state(model_name, optimizer_name, parameters, learning_rate)`
- `run_simple_training_loop_with_optimizer(weights, gradients, training_state)`

This module is where you go when you want neural-network-style helpers without building
everything manually.

### `undml.tensorforge`

This is the most framework-like part of the package. It gives you a small local API for
building models and optimizers.

Useful entry points:

- `create_forge_tensor_from_plain_values(values, requires_grad=False, name="tensor")`
- `create_forge_zero_tensor(shape, requires_grad=False, name="zeros")`
- `create_forge_one_tensor(shape, requires_grad=False, name="ones")`
- `create_forge_random_tensor(shape, minimum=-0.1, maximum=0.1, requires_grad=False, name="random")`
- `create_linear_layer(input_size, output_size, bias=True)`
- `create_sequential_model(*layers)`
- `create_relu_layer()`
- `create_sigmoid_layer()`
- `create_tanh_layer()`
- `create_softmax_layer(axis=-1)`
- `create_sgd_optimizer(parameters, learning_rate=0.01, momentum=0.0)`
- `create_adam_optimizer(parameters, learning_rate=0.001)`
- `create_mean_squared_error_loss(predicted, target)`
- `create_cross_entropy_loss(logits, target_index)`

The main classes are:

- `Parameter`
- `Module`
- `Linear`
- `Sequential`
- `ReLU`
- `Sigmoid`
- `Tanh`
- `Softmax`
- `SGD`
- `Adam`

This is the part to use when you want a compact model-building workflow.

### `undml.load`

This package is what the `.und` runtime uses to resolve imports.

Main function:

- `load_simp_module(module_name)`

It can load:

- normal Python modules
- local `.und` files
- compatibility aliases such as `tensor_matrix_library`, `advanced_ml_library`, and `forge_helpers`

## Quick examples

### Tensor math

```python
from undml.tensor.tensor_matrix_operation import Tensor

x = Tensor([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)
y = x.relu()
z = y.mean()
z.backward()
```

### Broadcasting

```python
from undml.tensor.tensor_matrix_operation import apply_tensor_broadcast_addition

result = apply_tensor_broadcast_addition([[1.0, 2.0], [3.0, 4.0]], [0.5, 0.5])
```

### Attention

```python
from undml.adv.advanced_ml_operation import create_scaled_dot_product_attention_matrix

output = create_scaled_dot_product_attention_matrix(
    [[1.0, 0.0]],
    [[1.0, 0.0], [0.0, 1.0]],
    [[10.0, 0.0], [0.0, 1.0]],
)
```

### Small model

```python
from undml.tensorforge.forge_helpers import create_linear_layer, create_sequential_model, create_relu_layer

model = create_sequential_model(
    create_linear_layer(2, 4),
    create_relu_layer(),
    create_linear_layer(4, 1),
)
```

## Compatibility notes

- The older `src/modules` imports are still available as compatibility wrappers.
- Existing `.und` files can still use the older library names like `tensor_matrix_library`
  and `advanced_ml_library`.
- The local loader also supports the new `forge_helpers` name for the framework-style API.

## Practical guidance

Use `tensor` when you need:

- raw tensor manipulation
- shape logic
- simple gradients
- broadcasting

Use `adv` when you need:

- loss functions
- optimizer state
- attention
- transformer-style helpers

Use `tensorforge` when you need:

- layer composition
- model objects
- optimizer objects
- a compact training workflow

If you want, I can also expand this README with a “mini tutorial” section showing a complete
forward pass, loss computation, and optimizer step using the new `tensorforge` API.
