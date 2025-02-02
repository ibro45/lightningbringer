# How to Create a Custom Metric in Lighter

## Introduction to Custom Metrics

Metrics are essential for evaluating the performance of your deep learning models during training, validation, and testing. Lighter seamlessly integrates with `torchmetrics`, a powerful library for defining and computing various metrics in PyTorch. While `torchmetrics` provides a wide range of built-in metrics, you may often need to create custom metrics tailored to your specific research problem or evaluation criteria.

This how-to guide will walk you through the process of defining and using custom metrics in Lighter. By creating custom metrics, you can gain deeper insights into your model's behavior and ensure that you are evaluating performance according to your specific needs.

## `torchmetrics` Basics for Custom Metrics

Lighter leverages `torchmetrics` as the foundation for its metric system. To create custom metrics in Lighter, you need to understand the basic concepts of `torchmetrics`:

*   **`Metric` Class**: In `torchmetrics`, a metric is defined as a class that inherits from `torchmetrics.Metric`. This base class provides the fundamental structure and methods for defining custom metrics.
*   **`update()` Method**: The `update()` method is a crucial part of a custom metric. It is called for each batch of data during training, validation, or testing. In the `update()` method, you accumulate the necessary statistics or intermediate values based on the model's predictions and the ground truth targets.
*   **`compute()` Method**: The `compute()` method is called at the end of each epoch (or at the end of validation/testing). It calculates the final metric value based on the accumulated statistics from the `update()` method.
*   **`add_state()`**: To store the accumulated statistics within your custom metric class, you use the `add_state()` method. This method registers state variables that will be automatically managed by `torchmetrics`, including handling of distributed computation and resetting states between epochs.

## Creating a Custom Metric: Step-by-Step

Let's walk through the steps of creating a custom metric in Lighter using `torchmetrics`. We'll create a simple example custom metric called `MyCustomMetric` for binary classification, which calculates a variation of accuracy.

**1. Subclass `torchmetrics.Metric`**:

First, create a new Python file (e.g., `my_project/metrics/my_custom_metric.py`) within your project to define your custom metric class. Start by importing `torchmetrics.Metric` and subclassing it:

```python title="my_project/metrics/my_custom_metric.py"
from torchmetrics import Metric
import torch

class MyCustomMetric(Metric):
    def __init__(self):
        super().__init__()
        # ... (state initialization will be added in the next step) ...

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # ... (update logic will be added in the next step) ...
        pass

    def compute(self):
        # ... (compute logic will be added in the next step) ...
        pass
```

**2. Initialize Metric State using `add_state()`**:

In the `__init__` method of your custom metric class, use `self.add_state()` to initialize the state variables that will store the accumulated statistics. For our `MyCustomMetric`, we'll need to track the number of correct predictions and the total number of predictions.

```python title="my_project/metrics/my_custom_metric.py"
from torchmetrics import Metric
import torch

class MyCustomMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum") # Track correct predictions
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")   # Track total predictions
```

*   **`self.add_state("correct", ...)`**: This line registers a state variable named `"correct"`.
    *   `default=torch.tensor(0)`: Initializes the state variable to a PyTorch tensor with the value 0.
    *   `dist_reduce_fx="sum"`: Specifies how to reduce this state variable across distributed processes (e.g., in multi-GPU training). `"sum"` indicates that the values from different processes should be summed. `torchmetrics` handles the distributed synchronization automatically.
*   **`self.add_state("total", ...)`**: Similarly, this line registers a state variable named `"total"` to track the total number of predictions.

**3. Implement the `update()` Method**:

The `update()` method is called for each batch of predictions and targets. In this method, you need to:

1.  Process the input `preds` (predictions) and `target` tensors to extract the necessary information for your metric calculation.
2.  Update the state variables (defined in `__init__` using `add_state()`) based on the processed predictions and targets.

For `MyCustomMetric`, we'll assume binary classification where `preds` are probabilities in the range [0, 1] and `target` are binary labels (0 or 1). We'll consider a prediction as "correct" if the predicted probability is greater than or equal to 0.5 when the target is 1, or less than 0.5 when the target is 0.

```python title="my_project/metrics/my_custom_metric.py"
from torchmetrics import Metric
import torch

class MyCustomMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # 1. Process inputs: Convert probabilities to binary predictions (0 or 1)
        binary_preds = (preds >= 0.5).int()

        # 2. Update state: Count correct predictions and total predictions
        self.correct += torch.sum(binary_preds == target) # Increment 'correct' state
        self.total += target.numel()                     # Increment 'total' state
```

*   **`binary_preds = (preds >= 0.5).int()`**: This line converts the probability predictions (`preds`) into binary predictions (0 or 1) by thresholding at 0.5.
*   **`self.correct += torch.sum(binary_preds == target)`**: This line calculates the number of correct predictions in the current batch by comparing `binary_preds` with `target` and summing up the `True` values (where predictions match targets). It then increments the `self.correct` state variable by this count.
*   **`self.total += target.numel()`**: This line calculates the total number of predictions in the current batch using `target.numel()` (number of elements in the `target` tensor) and increments the `self.total` state variable.

**4. Implement the `compute()` Method**:

The `compute()` method is called at the end of each epoch to calculate the final metric value based on the accumulated state variables. For `MyCustomMetric`, we'll calculate the custom accuracy as the ratio of `correct` predictions to `total` predictions.

```python title="my_project/metrics/my_custom_metric.py"
from torchmetrics import Metric
import torch

class MyCustomMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        binary_preds = (preds >= 0.5).int()
        self.correct += torch.sum(binary_preds == target)
        self.total += target.numel()

    def compute(self):
        # Calculate custom accuracy: correct / total
        return self.correct.float() / self.total # Compute metric value
```

*   **`return self.correct.float() / self.total`**: This line calculates the custom accuracy by dividing the total number of correct predictions (`self.correct`) by the total number of predictions (`self.total`). We cast `self.correct` to float to ensure floating-point division. The computed metric value is then returned.

**5. Integrate with Lighter Configuration**:

Once you have defined your custom metric class in `my_project/metrics/my_custom_metric.py`, you can integrate it into your Lighter configuration (`config.yaml`) to use it during training, validation, or testing.

**Example: Using Custom Metric in `config.yaml`**

```yaml title="config.yaml"
project: my_project/ # Project root directory

system:
  metrics:
    train: # Metrics to be computed during training
      - _target_: torchmetrics.Accuracy # Built-in Accuracy metric
      - _target_: my_project.metrics.MyCustomMetric # Custom metric

    val:   # Metrics to be computed during validation
      - _target_: torchmetrics.Accuracy
      - _target_: my_project.metrics.MyCustomMetric
```

In this example:

*   We specify the `project` path to tell Lighter where to find custom modules.
*   In the `system.metrics.train` and `system.metrics.val` sections, we define lists of metrics to be computed during training and validation, respectively.
*   `_target_: torchmetrics.Accuracy`: This line uses a built-in metric from `torchmetrics` (Accuracy).
*   `_target_: my_project.metrics.MyCustomMetric`: This line uses our custom metric `MyCustomMetric` defined in `my_project/metrics/my_custom_metric.py`. Lighter will dynamically load and instantiate this custom metric class.

Now, when you run your Lighter experiment (e.g., using `lighter fit config.yaml`), Lighter will automatically:

1.  Instantiate both the built-in `Accuracy` metric and your custom `MyCustomMetric`.
2.  Call the `update()` method of both metrics for each batch of data in the training and validation dataloaders.
3.  Call the `compute()` method of both metrics at the end of each epoch to calculate the final metric values.
4.  Log and display the computed values for both metrics during training and validation.

## Complete Custom Metric Example

Here's the complete code for our example custom metric, `MyCustomMetric` (in `my_project/metrics/my_custom_metric.py`):

```python title="my_project/metrics/my_custom_metric.py"
from torchmetrics import Metric
import torch

class MyCustomMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        binary_preds = (preds >= 0.5).int()
        self.correct += torch.sum(binary_preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
```

And here's how you would use it in your `config.yaml`:

```yaml title="config.yaml"
project: my_project/

system:
  metrics:
    train:
      - _target_: torchmetrics.Accuracy
      - _target_: my_project.metrics.MyCustomMetric

    val:
      - _target_: torchmetrics.Accuracy
      - _target_: my_project.metrics.MyCustomMetric
```

## Recap: Creating and Using Custom Metrics

Creating custom metrics in Lighter using `torchmetrics` involves these key steps:

1.  **Subclass `torchmetrics.Metric`**: Create a new class that inherits from `torchmetrics.Metric`.
2.  **Initialize State with `add_state()`**: In the `__init__` method, define and initialize state variables using `self.add_state()` to store accumulated statistics.
3.  **Implement `update()`**: Define the `update()` method to process predictions and targets for each batch and update the state variables accordingly.
4.  **Implement `compute()`**: Define the `compute()` method to calculate the final metric value based on the accumulated state variables.
5.  **Integrate in `config.yaml`**: Reference your custom metric class in the `system.metrics` section of your `config.yaml` file, using the `_target_` key and the module path to your custom metric class.

By following these steps, you can easily extend Lighter with your own custom metrics, enabling you to evaluate your deep learning models with metrics that are perfectly aligned with your research goals and evaluation criteria.

Next, explore the [How-To guide on Using Adapters](04_using_adapters.md) to learn how to customize data handling and argument passing in Lighter, or return to the [How-To guides section](../how-to/) for more practical problem-solving guides. You can also go back to the [Explanation section](../explanation/) for more conceptual documentation or the [Tutorials section](../tutorials/) for end-to-end examples.
