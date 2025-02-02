# How to Use Adapters in Lighter

## Introduction to Adapters

Adapters are a powerful feature in Lighter that provide a flexible way to customize data handling and argument passing within your deep learning experiments. They act as intermediaries, allowing you to adapt the flow of data and arguments between different components of Lighter without modifying the core framework itself.

This how-to guide will explain the purpose and usage of adapters in Lighter. By understanding and utilizing adapters, you can tailor Lighter to seamlessly integrate with diverse data formats, model architectures, and custom logic, making your experiments more adaptable and maintainable.

## Types of Adapters in Lighter

Lighter provides several built-in adapter types, each designed to address specific customization needs:

1.  **`BatchAdapter`**:

    *   **Purpose**: The `BatchAdapter` is responsible for extracting the essential components from a data batch: `input`, `target`, and `identifier`. It standardizes the batch structure, ensuring that Lighter's core components (like loss functions and metrics) receive data in a consistent format, regardless of the original batch structure from your dataloaders.
    *   **Use Cases**:
        *   **Handling Complex Batch Structures**: When your dataloaders produce batches with complex structures (e.g., nested dictionaries, lists of tuples, multi-modal data), `BatchAdapter` allows you to define accessors to extract the relevant `input`, `target`, and `identifier` tensors.
        *   **Data Format Conversion**: If your data needs to be rearranged or converted into a specific format before being passed to the model or loss function, you can implement this logic within a custom `BatchAdapter` or use callable accessors for simple transformations.

    *   **Configuration**: You configure `BatchAdapter` in your `config.yaml` by specifying accessors for `input`, `target`, and `identifier`. Accessors can be:
        *   **Integer Index**: For list or tuple batches, use an integer index (e.g., `0`, `1`) to access elements by position.
        *   **String Key**: For dictionary batches, use a string key (e.g., `"image"`, `"label"`) to access elements by key.
        *   **Callable Function**: For more complex batch structures or transformations, provide a callable function that takes the batch as input and returns the desired `input`, `target`, or `identifier`.

    *   **Example**:

        ```yaml title="config.yaml"
        system:
          adapters:
            train: # Adapters for the training stage
              batch: # BatchAdapter configuration
                _target_: lighter.adapters.BatchAdapter # Use the built-in BatchAdapter
                input_accessor: "image"              # Access input using key "image"
                target_accessor: "label"             # Access target using key "label"
                identifier_accessor: "id"            # Access identifier using key "id"
        ```

        In this example, we configure the `BatchAdapter` for the training stage (`train`). We specify that the `input` should be accessed using the key `"image"`, the `target` using the key `"label"`, and the `identifier` using the key `"id"` from the input batch (assuming the batch is a dictionary).

2.  **`CriterionAdapter`**:

    *   **Purpose**: The `CriterionAdapter` customizes how arguments are passed to your criterion (loss function) and allows you to apply transformations to the `input`, `target`, and `pred` (prediction) tensors before they are passed to the criterion.
    *   **Use Cases**:
        *   **Adapting Argument Order**: If your criterion function expects arguments in a different order than the default `(pred, target)` order used by Lighter, `CriterionAdapter` lets you rearrange them.
        *   **Applying Pre-processing Transforms**: You can apply pre-processing transforms to `pred`, `target`, or `input` tensors before passing them to the criterion. This is useful for tasks like:
            *   Converting predictions to probabilities (e.g., using `torch.softmax`).
            *   Applying one-hot encoding to targets.
            *   Reshaping tensors to match the criterion's input requirements.
        *   **Selecting Specific Arguments**: If your criterion function accepts only specific arguments (e.g., only `pred` and `target`, but not `input`), you can use `CriterionAdapter` to filter and select the relevant arguments.

    *   **Configuration**: You configure `CriterionAdapter` in your `config.yaml` by specifying:
        *   **Argument Positions or Names**: Use `pred_argument`, `target_argument`, and `input_argument` to define how `pred`, `target`, and `input` tensors should be mapped to the arguments of your criterion function. You can specify either positional arguments (integers starting from 0) or keyword argument names (strings).
        *   **Transforms**: Use `pred_transforms`, `target_transforms`, and `input_transforms` to define lists of transforms to be applied to `pred`, `target`, and `input` tensors, respectively, before they are passed to the criterion. Transforms are specified using the MONAI Bundle syntax (e.g., `_target_: torch.argmax`).

    *   **Example**:

        ```yaml title="config.yaml"
        system:
          adapters:
            train: # Adapters for the training stage
              criterion: # CriterionAdapter configuration
                _target_: lighter.adapters.CriterionAdapter # Use the built-in CriterionAdapter
                pred_argument: 1                      # Map 'pred' to the 2nd positional argument (index 1)
                target_argument: 0                    # Map 'target' to the 1st positional argument (index 0)
                pred_transforms:                      # Apply transforms to 'pred'
                  - _target_: torch.sigmoid            # Apply sigmoid activation to predictions
        ```

        In this example, we configure the `CriterionAdapter` for the training stage (`train`). We specify that the `target` tensor should be passed as the first positional argument (index 0) to the criterion, and the `pred` tensor as the second positional argument (index 1). We also apply a sigmoid activation function to the `pred` tensor before passing it to the criterion, using `pred_transforms`.

3.  **`MetricsAdapter`**:

    *   **Purpose**: The `MetricsAdapter` is similar to `CriterionAdapter`, but it is used to customize argument passing and apply transforms for your metrics. It controls how `input`, `target`, and `pred` tensors are passed to your metric functions or classes.
    *   **Use Cases**:
        *   **Adapting Metric Argument Order**: If your metric functions or classes expect arguments in a specific order, `MetricsAdapter` allows you to rearrange them.
        *   **Applying Metric-Specific Transforms**: You can apply transforms that are specific to your metrics, such as:
            *   Converting predictions to class labels (e.g., using `torch.argmax`).
            *   Applying thresholding or other pre-processing steps required by your metrics.
        *   **Filtering Arguments for Metrics**: If your metrics only need a subset of `input`, `target`, and `pred` tensors, you can use `MetricsAdapter` to filter and pass only the necessary arguments.

    *   **Configuration**: The configuration of `MetricsAdapter` is identical to `CriterionAdapter`. You use `pred_argument`, `target_argument`, `input_argument` to map arguments and `pred_transforms`, `target_transforms`, `input_transforms` to apply transforms, just like in `CriterionAdapter`.

    *   **Example**:

        ```yaml title="config.yaml"
        system:
          adapters:
            val: # Adapters for the validation stage
              metrics: # MetricsAdapter configuration
                _target_: lighter.adapters.MetricsAdapter # Use the built-in MetricsAdapter
                pred_argument: "prediction"           # Map 'pred' to keyword argument "prediction"
                target_argument: "ground_truth"         # Map 'target' to keyword argument "ground_truth"
                pred_transforms:                      # Apply transforms to 'pred'
                  - _target_: torch.argmax              # Convert predictions to class labels
                    dim: 1
        ```

        In this example, we configure the `MetricsAdapter` for the validation stage (`val`). We specify that the `pred` tensor should be passed to the metric as a keyword argument named `"prediction"`, and the `target` tensor as a keyword argument named `"ground_truth"`. We also apply a transform to convert the predictions to class labels using `torch.argmax` before passing them to the metric.

4.  **`LoggingAdapter`**:

    *   **Purpose**: The `LoggingAdapter` is specifically designed to customize the data that is logged by Lighter during training, validation, and testing. It allows you to apply transforms to `input`, `target`, and `pred` tensors right before they are logged, enabling you to format, filter, or pre-process the data for better logging and visualization.
    *   **Use Cases**:
        *   **Formatting Logged Data**: You can format tensors for better readability in logs, such as converting tensors to NumPy arrays, detaching them from the computation graph, or converting numerical tensors to images or other visual representations.
        *   **Filtering Logged Data**: If you only want to log specific parts of the `input`, `target`, or `pred` tensors, you can use `LoggingAdapter` to filter and select the data to be logged.
        *   **Applying Logging-Specific Transforms**: You can apply transforms that are specifically intended for logging, such as:
            *   Converting one-hot encoded tensors back to class labels.
            *   Scaling or normalizing tensors for visualization.
            *   Creating visualizations (e.g., images, histograms) from tensors to be logged as media.

    *   **Configuration**: You configure `LoggingAdapter` in your `config.yaml` using `pred_transforms`, `target_transforms`, and `input_transforms` to define lists of transforms to be applied to `pred`, `target`, and `input` tensors, respectively, before logging.

    *   **Example**:

        ```yaml title="config.yaml"
        system:
          adapters:
            train: # Adapters for the training stage
              logging: # LoggingAdapter configuration
                _target_: lighter.adapters.LoggingAdapter # Use the built-in LoggingAdapter
                input_transforms:                       # Apply transforms to 'input' before logging
                  - _target_: monai.transforms.ToNumpy  # Convert input tensor to NumPy array
                pred_transforms:                        # Apply transforms to 'pred' before logging
                  - _target_: torch.argmax                # Convert predictions to class labels
                    dim: 1
        ```

        In this example, we configure the `LoggingAdapter` for the training stage (`train`). We apply transforms to both `input` and `pred` tensors before they are logged. We convert the `input` tensor to a NumPy array using `monai.transforms.ToNumpy` and convert the `pred` tensor to class labels using `torch.argmax`. This ensures that the logged data is in a more human-readable and log-friendly format.

## Creating Custom Adapters (Advanced)

While Lighter provides built-in adapters for common customization needs, you may encounter scenarios where you require more specialized adapter logic. In such cases, you can create your own custom adapter classes by subclassing the base adapter classes provided by Lighter:

*   `lighter.adapters._ArgumentsAdapter`
*   `lighter.adapters._TransformsAdapter`
*   `lighter.adapters._ArgumentsAndTransformsAdapter`
*   `lighter.adapters.BatchAdapter`
*   `lighter.adapters._LoggingAdapter`

To create a custom adapter:

1.  **Subclass an Adapter Class**: Create a new Python file (e.g., `my_project/adapters/my_adapter.py`) and define your custom adapter class by subclassing one of the base adapter classes from `lighter.adapters`.
2.  **Implement Custom Logic**: Override the relevant methods (e.g., `__call__`, `update`, `compute`, `_transform`, `_access_value`) in your custom adapter class to implement your desired customization logic.
3.  **Integrate in `config.yaml`**: Reference your custom adapter class in your `config.yaml` file, using the `_target_` key and the module path to your custom adapter class, just like you would for built-in adapters.

Refer to the Lighter codebase (specifically the `lighter/adapters.py` file) for examples and detailed guidance on how to implement custom adapter classes.

## Recap: Unleashing Flexibility with Adapters

Adapters are a cornerstone of Lighter's flexibility and extensibility. By using adapters effectively, you can:

*   Seamlessly integrate Lighter with diverse data formats and batch structures.
*   Customize argument passing to loss functions and metrics.
*   Apply pre-processing and post-processing transforms at various stages of your experiment.
*   Tailor the data that is logged for monitoring and analysis.
*   Create highly specialized and reusable customization components.

Mastering adapters empowers you to adapt Lighter to your specific research needs and build complex, yet well-organized and maintainable deep learning experiment configurations.

Next, explore the [How-To guide on Using and Extending Writers](05_using_and_extending_writers.md) to learn how to save model predictions and outputs to files, or return to the [How-To guides section](../how-to/) for more practical problem-solving guides. You can also go back to the [Explanation section](../explanation/) for more conceptual documentation or the [Tutorials section](../tutorials/) for end-to-end examples.
