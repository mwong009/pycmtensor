# User guide

---

## Data structures

PyCMTensor works by specifying utility equations, variables, indices, and outputs using [symbolic tensors](https://blog.tensorflow.org/2019/01/what-are-symbolic-and-imperative-apis.html), similar to programs like TensorFlow. 
These do not hold any value until an input is given. 
The entire model is build as a computational graph with nodes representing intermediate values and edges representing mathematical operations. 
The graph is then manipulated internally and optimized during compilation.
When an input is provided to certain tensors, the graph computes the values for other symbols in the graph. 

The core API library, pytensor, gives us a set of tools that takes the defined symbolic tensors, constructs a computational graph and optimizes the mathematical operations around it.

As a developer, you define or extend a new model by creating new tensor variable objects from primitive tensors. For instance, take a basic linear equation, $y = bx +c$. The variables $y$, $x$, and $c$ are symbolic tensors making up a computational graph, and $b$ is a shared variable consisting of a mutable defined value. When we call `train(y)`, an pytensor function is executed on the equation and an update value is passed to $b$.

A PyCMTensor model is composed of 

1. A Database object holding the symbolic tensors of input and choice variables
2. Beta variables, which are the taste parameters in the choice model to be estimated
   - Optionally, Random (symbolic) variables, to define random components in the model
3. Neural networks, defined as layers with inputs and outputs, and
4. A function which translates a set of utility equations into a symbolic tensor output

**Additional complexity**

This symbolic approach does add complexity, however, the tradeoffs are that a. more complex structures can be defined more easily and with internal code optimizations, this can result in very efficient estimation of very large datasets. b. The most compelling use of computational graph approach is that for econometric analysis, we can inspect any part of the graph and obtain statistics on any part of the graph. This is especially practical for neural network based discrete choice models.

### `Beta`

A `Beta` variable is the main data object when estimate a PyCMTensor model. `Beta` inherits the `TensorExpression` class which adds pytensor tensor operations to the data object. A `Beta` is defined as such:

```python
Beta(name, value, lb, ub, status)
```

The `Beta` object can be maniputated in various ways:

In mathematical operations, for instance:

```python
from pycmtensor.expressions import Beta
import pytensor.tensor as aet
import numpy as np 
beta_cost = Beta("beta_cost", 2.)
train_cost = aet.vector("train_cost") 
y = beta_cost * train_cost
y.eval({train_cost: np.array([1, 2, 3])})
```

Output:
```console
array([2., 4., 6.])
```

`train_cost` and `y` are symbolic variables, while `beta_cost` is a `TensorExpression.Beta` object




**Updating values**

During model estimation, a global optimization function is called (e.g. log likelihood) and updates are received through the gradients with respect to the cost function. The `Beta` is then updated with this new value, and the estimation run iteratively, until convergence is reached.


**Convergence**

Model convergence is defined as reaching the minimum gradient norm of the model parameters. For non-convex optimizations, for example: large, complex neural network models, convergence may not be reached, therefore many implementations are added such as learning rate adaptation algorithms and learning rate decay are avaiable so that sufficient global convergence can be achieved, albeit non-optimally.  
 

## Calling variables and selecting data

## Mathematical operations 

## Working with generic and alternative specific variables

## Model estimation

### Configuration

### Optimizers

## Fine tuning model

## Generate results