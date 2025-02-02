# How to Implement a Custom Inferer in Lighter

## Introduction to Inferers

Inference, in the context of deep learning, is the process of using a trained model to make predictions on new, unseen data. In Lighter, *inferers* are responsible for defining how this inference process is carried out during validation, testing, and prediction stages.

Inferers handle tasks such as:

*   **Sliding Window Inference**: For processing large images or volumes that don't fit into memory at once.
*   **Test-Time Augmentation (TTA)**: Applying augmentations to input data at test time and aggregating predictions.
*   **Model Ensembling**: Combining predictions from multiple models.
*   **Custom Output Processing**: Applying custom logic to model outputs before they are returned or saved.

Lighter leverages MONAI's powerful inferer implementations as a starting point and allows you to easily use and customize them. Furthermore, you can implement your own custom inferers to handle specific inference requirements.

This how-to guide will explain how to implement and use custom inferers in Lighter, giving you full control over the inference process in your deep learning workflows.

## Using MONAI Inferers in Lighter

Lighter seamlessly integrates with MONAI's inferers, providing a range of pre-built inferer implementations that you can readily use in your projects.

**Configuration**:

You configure the inferer in your `config.yaml` within the `system.inferer` section:

```yaml title="config.yaml"
system:
  inferer:
    _target_: monai.inferers.SlidingWindowInferer # Use MONAI's SlidingWindowInferer
    roi_size: [96, 96, 96]                     # Region of interest size for sliding window
    sw_batch_size: 4                           # Batch size for sliding window inference
    overlap: 0.5                               # Overlap ratio between windows
```

*   **`_target_: monai.inferers.SlidingWindowInferer`**: Specifies that you want to use MONAI's `SlidingWindowInferer`.
*   **`roi_size`, `sw_batch_size`, `overlap`**: These are arguments specific to `SlidingWindowInferer`. Refer to MONAI documentation for details on available inferers and their arguments.

**Commonly Used MONAI Inferers**:

*   **`monai.inferers.SlidingWindowInferer`**: For sliding window inference, useful for processing large images or volumes.
*   **`monai.inferers.SimpleInferer`**: A basic inferer that directly feeds the entire input tensor to the model. Suitable for cases where inputs fit in memory.
*   **`monai.inferers.EnsembleInferer`**: For ensembling predictions from multiple models.
*   **`monai.inferers.patch_inferer`**: For patch-based inference.

To use a MONAI inferer, simply specify its `_target_` in your `config.yaml` and provide the necessary arguments as documented in MONAI.

**Example: Using `SlidingWindowInferer` for Validation**

```yaml title="config.yaml"
system:
  model: # ... (model definition) ...
  inferer:
    _target_: monai.inferers.SlidingWindowInferer # Use SlidingWindowInferer
    roi_size: [128, 128, 128]
    sw_batch_size: 8
    overlap: 0.25

  def validation_step(self, batch, batch_idx):
    output = super().validation_step(batch, batch_idx) # Call base validation step
    # Inference is automatically handled by the configured inferer
    pred = output[Data.PRED] # Get predictions (already processed by inferer)
    # ... (rest of validation step logic) ...
```

In this example, we configure `SlidingWindowInferer` in the `system.inferer` section. During the validation stage, Lighter will automatically use this inferer in the `forward` pass of your `System` (in `system.py`). When `self.forward(input)` is called within `validation_step`, if an inferer is configured and the current mode is validation, testing, or prediction, Lighter will use the inferer to process the input and obtain predictions.

## Implementing a Custom Inferer

While MONAI provides a rich set of inferers, you might need to implement a custom inferer for specialized inference logic, such as:

*   **Test-Time Augmentation (TTA)**: To apply augmentations during inference.
*   **Advanced Ensembling Strategies**: Beyond simple averaging of predictions.
*   **Highly Specialized Output Processing**: Unique to your research problem.

To implement a custom inferer in Lighter, you need to create a Python class that follows a specific structure.

**Custom Inferer Class Structure**:

A custom inferer class should have the following structure:

```python title="my_project/inferers/my_custom_inferer.py"
import torch
import monai

class MyCustomInferer:
    def __init__(self, arg1, arg2, **kwargs):
        """
        Initialize your custom inferer.

        Args:
            arg1: Custom argument 1.
            arg2: Custom argument 2.
            **kwargs: Additional keyword arguments.
        """
        self.arg1 = arg1
        self.arg2 = arg2
        # ... (initialize any internal components) ...

    def __call__(self, inputs: torch.Tensor, network: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """
        Perform inference using your custom logic.

        Args:
            inputs: Input tensor(s) to the model.
            network: The deep learning model (torch.nn.Module).
            *args: Additional positional arguments (if needed).
            **kwargs: Additional keyword arguments (if needed).

        Returns:
            The processed prediction tensor(s).
        """
        # 1. Custom Inference Logic: Implement your specialized inference steps here.
        #    This might include:
        #    - Test-time augmentation (TTA)
        #    - Model ensembling
        #    - Sliding window or patch-based inference (potentially using MONAI's inferers internally)
        #    - Any other custom processing steps

        # Example: Basic inference (no custom logic)
        outputs = network(inputs) # Forward pass through the model
        processed_outputs = self.post_process(outputs) # Apply post-processing if needed
        return processed_outputs

    def post_process(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Optional post-processing step for model outputs.

        Args:
            outputs: Raw model output tensor(s).

        Returns:
            Processed output tensor(s).
        """
        # Implement any post-processing steps here (e.g., thresholding, softmax, etc.)
        return outputs # Return processed outputs
```

**Key Components of a Custom Inferer Class**:

1.  **`__init__(self, *args, **kwargs)`**:
    *   The constructor of your inferer class.
    *   Takes custom arguments that you can configure in your `config.yaml`.
    *   Initialize any internal components or parameters needed for your inference logic.

2.  **`__call__(self, inputs: torch.Tensor, network: torch.nn.Module, *args, **kwargs) -> torch.Tensor`**:
    *   Makes your inferer class callable like a function. This is the main inference method.
    *   **Arguments**:
        *   `inputs` (`torch.Tensor`): The input tensor(s) to your model.
        *   `network` (`torch.nn.Module`): The deep learning model (`self.model` in your `System`).
        *   `*args`, `**kwargs`:  Allows for passing additional arguments if needed (though not commonly used).
    *   **Logic**:
        *   Implement your custom inference logic within this method. This is where you define how inference is performed.
        *   You will typically perform a forward pass through the `network` (model) using `outputs = network(inputs)`.
        *   You can incorporate various inference techniques here (TTA, ensembling, sliding window, etc.).
        *   Optionally call a `post_process` method to further process the raw model outputs.
    *   **Return Value**:
        *   Return the processed prediction tensor(s) as a `torch.Tensor`. This tensor will be used as the `pred` output in your validation, testing, or prediction steps.

3.  **`post_process(self, outputs: torch.Tensor) -> torch.Tensor` (Optional)**:
    *   An optional method for applying post-processing steps to the raw model outputs.
    *   This can include operations like thresholding, applying a softmax function, or any other task-specific processing.
    *   If you don't need post-processing, you can simply return the `outputs` tensor as is.

**Example: Custom Inferer for Test-Time Augmentation (TTA)**

```python title="my_project/inferers/tta_inferer.py"
import torch
import monai
import copy

class TestTimeAugmentationInferer:
    def __init__(self, augmentations, aggregation_mode="mean"):
        """
        Initialize TestTimeAugmentationInferer.

        Args:
            augmentations (list[Callable]): List of augmentation transforms to apply during TTA.
            aggregation_mode (str): Method to aggregate predictions (e.g., "mean", "median").
        """
        self.augmentations = augmentations
        self.aggregation_mode = aggregation_mode

    def __call__(self, inputs: torch.Tensor, network: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """
        Perform test-time augmentation inference.

        Args:
            inputs: Input tensor(s).
            network: The model.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Aggregated prediction tensor.
        """
        all_predictions = []

        # 1. Original Prediction (no augmentation)
        original_prediction = network(inputs)
        all_predictions.append(original_prediction)

        # 2. Augmented Predictions
        for aug in self.augmentations:
            augmented_input = aug(inputs) # Apply augmentation
            prediction = network(augmented_input) # Inference on augmented input

            # Inverse augment prediction (if necessary, for spatial augmentations)
            inverse_augmented_prediction = self.inverse_augment(prediction, aug)
            all_predictions.append(inverse_augmented_prediction)

        # 3. Aggregate Predictions
        aggregated_prediction = self.aggregate_predictions(all_predictions, mode=self.aggregation_mode)
        return aggregated_prediction

    def inverse_augment(self, prediction, augmentation):
        """
        Inverse augment the prediction tensor to align with the original input space.

        Args:
            prediction: Prediction tensor from augmented input.
            augmentation: Augmentation transform applied.

        Returns:
            Inverse augmented prediction tensor.
        """
        # Implement inverse augmentation logic if augmentations are spatial (e.g., rotations, flips)
        # For simple augmentations like intensity scaling, inverse augmentation might not be needed.
        return prediction # Return prediction as is (no inverse augmentation in this example)

    def aggregate_predictions(self, predictions, mode="mean"):
        """
        Aggregate multiple prediction tensors into a single tensor.

        Args:
            predictions (list[torch.Tensor]): List of prediction tensors to aggregate.
            mode (str): Aggregation mode ("mean", "median", etc.).

        Returns:
            Aggregated prediction tensor.
        """
        if mode == "mean":
            return torch.mean(torch.stack(predictions), dim=0) # Mean aggregation
        elif mode == "median":
            return torch.median(torch.stack(predictions), dim=0).values # Median aggregation
        else:
            raise ValueError(f"Unsupported aggregation mode: {mode}")
```

In this `TestTimeAugmentationInferer` example:

*   The `__init__` method takes a list of augmentation transforms and an aggregation mode as arguments.
*   The `__call__` method performs TTA:
    1.  It gets the original prediction without augmentation.
    2.  It iterates through the provided augmentations, applies each augmentation to the input, gets predictions on augmented inputs, and (optionally) inverse-augments the predictions.
    3.  It aggregates all predictions (original + augmented) using the specified `aggregation_mode` (e.g., "mean").
*   `inverse_augment` method (in this example, it's a placeholder - you'd implement actual inverse augmentation logic if your augmentations are spatial).
*   `aggregate_predictions` method aggregates the list of predictions using mean or median.

**Integrating Custom Inferer with Lighter**:

1.  **Save your custom inferer class** (e.g., `TestTimeAugmentationInferer`) in a Python file within your project (e.g., `my_project/inferers/tta_inferer.py`).
2.  **Configure in `config.yaml`**: Specify your custom inferer in the `system.inferer` section of your `config.yaml` file, providing the path to your class and any arguments required by its `__init__` method:

    ```yaml title="config.yaml"
    system:
      inferer:
        _target_: my_project.inferers.tta_inferer.TestTimeAugmentationInferer # Path to custom inferer class
        augmentations: # Arguments for custom inferer's __init__
          - _target_: monai.transforms.RandFlipd
            keys: "image"
            prob: 0.5
            spatial_axis: [0]
          - _target_: monai.transforms.RandRotate90d
            keys: "image"
            prob: 0.5
            max_k: 2
        aggregation_mode: "mean"
    ```

    *   **`_target_: my_project.inferers.tta_inferer.TestTimeAugmentationInferer`**: Specifies the path to your custom inferer class.
    *   **`augmentations`, `aggregation_mode`**: These are arguments passed to the `__init__` method of your `TestTimeAugmentationInferer` class. In this case, we configure a list of MONAI augmentation transforms and the aggregation mode.

With this configuration, Lighter will instantiate your `TestTimeAugmentationInferer` and use it for inference during validation, testing, and prediction stages.

## Recap: Flexible Inference with Custom Inferers

Implementing custom inferers in Lighter empowers you to go beyond standard inference techniques and incorporate specialized logic tailored to your research needs. Whether it's test-time augmentation, advanced ensembling, or highly customized output processing, custom inferers provide the flexibility to define and execute inference in a way that best suits your deep learning tasks. By combining custom inferers with Lighter's configuration-driven approach, you can create sophisticated and reproducible inference workflows.

Next, explore the [Explanation section on Adapter System](../explanation/03_adapter_system.md) to understand how adapters further enhance customization in Lighter, or return to the [How-To guides section](../how-to/) for more practical problem-solving guides. You can also go back to the [Explanation section](../explanation/) for more conceptual documentation or the [Tutorials section](../tutorials/) for end-to-end examples.
