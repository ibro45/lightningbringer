# How to Use and Extend Writers in Lighter

## Introduction to Writers

Writers in Lighter are callbacks that handle the task of saving model predictions and other relevant outputs to files during validation, testing, and prediction stages. They provide a standardized and extensible way to persist your experiment results for analysis, visualization, and further use.

Lighter offers two main types of writers:

1.  **`FileWriter`**: For saving individual tensors (e.g., prediction maps, images) to separate files in various formats (e.g., NIfTI, NRRD, PNG, MP4, raw tensors).
2.  **`TableWriter`**: For saving tabular data (e.g., metrics, aggregated predictions) to CSV files.

This how-to guide will explain how to use and extend both `FileWriter` and `TableWriter` in Lighter. By mastering writers, you can effectively manage and utilize the outputs of your deep learning experiments.

## Using `FileWriter`

The `FileWriter` callback is designed to save individual tensors to files. It supports a variety of output formats and is highly customizable.

**Configuration**:

You configure `FileWriter` in your `config.yaml` within the `trainer.callbacks` section:

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter # Use the FileWriter callback
      path: "outputs/predictions"          # Directory to save output files
      writer: "itk_nifti"                  # Writer function to use (ITK-NIfTI format)
```

*   **`_target_: lighter.callbacks.FileWriter`**: Specifies that you want to use the `FileWriter` callback.
*   **`path: "outputs/predictions"`**: Defines the directory where the output files will be saved. Lighter will create this directory if it doesn't exist.
*   **`writer: "itk_nifti"`**: Specifies the writer function to be used for saving tensors. In this example, we use `"itk_nifti"`, which saves tensors in the ITK-NIfTI format (commonly used for medical images).

**Built-in Writer Functions**:

`FileWriter` comes with several built-in writer functions, each supporting a different output format:

*   **`"tensor"`**: Saves tensors as raw NumPy `.npy` files. Suitable for general-purpose tensor saving.
*   **`"image"`**: Saves 2D or 3D tensors as image files (PNG format for 2D, animated MP4 for 3D). Useful for saving image-like predictions or visualizations.
*   **`"video"`**: Saves 4D tensors (e.g., time-series of 3D volumes) as video files (MP4 format). Designed for saving video or animation outputs.
*   **`"itk_nrrd"`**: Saves 2D or 3D tensors as NRRD files (using ITK library). A common format in medical imaging.
*   **`"itk_seg_nrrd"`**: Saves segmentation mask tensors as NRRD files (using ITK library). Optimized for segmentation masks.
*   **`"itk_nifti"`**: Saves 2D or 3D tensors as NIfTI files (using ITK library). Another popular format in medical imaging.

**Usage**:

Once `FileWriter` is configured in your `config.yaml`, Lighter automatically uses it during validation, testing, and prediction stages (if these stages are enabled in your configuration).

During these stages, for each batch of data, `FileWriter` will:

1.  Receive the `pred` (prediction) tensor from your system's `predict_step`, `validation_step`, or `test_step`.
2.  Optionally apply any transforms defined in your `LoggingAdapter` (if configured).
3.  Use the specified writer function (e.g., `"itk_nifti"`) to save the `pred` tensor to a file within the specified `path` directory.
4.  Name the file using the `identifier` from the batch (if available) or generate a unique filename.

**Example: Saving Segmentation Predictions in NIfTI Format**

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter
      path: "outputs/segmentations"
      writer: "itk_nifti" # Save as NIfTI files

system:
  # ... (other system configurations) ...
  dataloaders:
    val:
      dataset:
        _target_: monai.datasets.DecathlonDataset
        task: "Task09_Spleen"
        root: "data/"
        section: "validation"
        transform: # ... (data transforms) ...
      batch_size: 1
```

In this example, we configure `FileWriter` to save segmentation predictions during the validation stage. The predictions will be saved as NIfTI files in the `outputs/segmentations` directory. The filenames will be based on the identifiers from the validation dataset (e.g., patient IDs).

## Extending `FileWriter` with Custom Writers

If the built-in writer functions don't meet your needs, you can easily extend `FileWriter` by creating your own custom writer functions or classes.

**1. Create a Custom Writer Function**:

You can define a custom writer function that takes two arguments:

*   `path`: The full file path (including filename and extension) where the tensor should be saved.
*   `tensor`: The PyTorch tensor to be saved.

**Example: Custom Writer Function to Save Tensors as Text Files**

```python title="my_project/writers/my_custom_writer.py"
import torch
import numpy as np

def write_tensor_as_text(path: str, tensor: torch.Tensor):
    """
    Saves a PyTorch tensor to a text file.

    Args:
        path: The file path to save the tensor to (e.g., "outputs/my_tensor.txt").
        tensor: The PyTorch tensor to be saved.
    """
    tensor_numpy = tensor.cpu().numpy() # Convert to NumPy array
    np.savetxt(path, tensor_numpy)      # Save as text file
```

**2. Register the Custom Writer Function**:

To use your custom writer function with `FileWriter`, you need to register it in your `config.yaml` by providing the path to your Python module and the name of your writer function using the `writer` argument:

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter
      path: "outputs/text_tensors"
      writer: my_project.writers.my_custom_writer.write_tensor_as_text # Path to custom writer
```

*   **`writer: my_project.writers.my_custom_writer.write_tensor_as_text`**: This line specifies the path to your custom writer function. Lighter will dynamically import and use this function. Make sure to replace `"my_project.writers.my_custom_writer"` with the actual path to your module.

**3. Create a Custom Writer Class (Advanced)**:

For more complex writing logic or if you need to maintain state within your writer, you can create a custom writer class that inherits from `lighter.callbacks.writer.BaseWriter`.

**Example: Custom Writer Class to Save Tensors with Metadata**

```python title="my_project/writers/my_custom_writer_class.py"
from lighter.callbacks.writer import BaseWriter
import torch
import json
import os

class MyCustomClassWriter(BaseWriter):
    @property
    def writers(self):
        return {"tensor_with_metadata": self.write_tensor_with_metadata} # Register writer function

    def write(self, tensor: torch.Tensor, identifier: str):
        """
        Main write method called by FileWriter.
        """
        path = os.path.join(self.path, f"{identifier}.json") # Define output path
        self.write_tensor_with_metadata(path, tensor, identifier=identifier) # Call writer function

    def write_tensor_with_metadata(self, path: str, tensor: torch.Tensor, identifier: str):
        """
        Saves a tensor to a JSON file along with metadata.

        Args:
            path: The file path to save the JSON file to.
            tensor: The PyTorch tensor to be saved.
            identifier: An identifier for the tensor (e.g., filename or index).
        """
        metadata = {
            "identifier": identifier,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "timestamp": datetime.datetime.now().isoformat()
        }
        data = {
            "metadata": metadata,
            "data": tensor.cpu().numpy().tolist() # Convert tensor data to list
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4) # Save data and metadata to JSON file
```

In this example, `MyCustomClassWriter` class:

*   Inherits from `BaseWriter`.
*   Defines a custom writer function `write_tensor_with_metadata` that saves tensors to JSON files along with metadata (shape, dtype, timestamp).
*   Registers this custom writer function in the `writers` property, associating it with the key `"tensor_with_metadata"`.
*   Overrides the `write` method to define how filenames are generated and how the custom writer function is called.

**4. Use the Custom Writer Class in `config.yaml`**:

To use your custom writer class, specify the path to your class module and the registered writer key (from the `writers` property) in your `config.yaml`:

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter
      path: "outputs/metadata_tensors"
      writer: my_project.writers.my_custom_writer_class.MyCustomClassWriter.tensor_with_metadata # Custom class writer
```

*   **`writer: my_project.writers.my_custom_writer_class.MyCustomClassWriter.tensor_with_metadata`**: This line specifies the path to your custom writer class and the writer key (`"tensor_with_metadata"`) that you registered in the `writers` property of your class.

## Using `TableWriter`

The `TableWriter` callback is used to save tabular data to CSV files. It is particularly useful for logging metrics, aggregated predictions, or any data that can be represented in a table format.

**Configuration**:

Configure `TableWriter` in your `config.yaml` within the `trainer.callbacks` section:

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.TableWriter # Use the TableWriter callback
      path: "outputs/metrics.csv"          # Path to save the CSV file
```

*   **`_target_: lighter.callbacks.TableWriter`**: Specifies that you want to use the `TableWriter` callback.
*   **`path: "outputs/metrics.csv"`**: Defines the path to the CSV file where the tabular data will be saved.

**Usage**:

To use `TableWriter`, you need to return a dictionary from your system's `validation_step`, `test_step`, or `predict_step` methods. `TableWriter` will automatically extract the key-value pairs from this dictionary and save them as rows in the CSV file.

**Example: Logging Metrics to CSV using `TableWriter`**

```python title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.TableWriter
      path: "outputs/metrics.csv" # Save metrics to CSV

system:
  metrics:
    val:
      - _target_: torchmetrics.Accuracy
      - _target_: torchmetrics.DiceCoefficient

  def validation_step(self, batch, batch_idx):
    output = super().validation_step(batch, batch_idx) # Call base validation step
    metrics = output[Data.METRICS] # Get computed metrics
    self.log_dict(metrics)        # Log metrics for display
    return metrics                # Return metrics dictionary for TableWriter
```

In this example:

*   We configure `TableWriter` to save data to `outputs/metrics.csv`.
*   In the `validation_step` of our `System`, we:
    *   Call the base class `validation_step` to perform the standard validation logic and compute metrics.
    *   Extract the computed metrics from the `output` dictionary.
    *   Return the `metrics` dictionary from `validation_step`.

    `TableWriter` will automatically capture the dictionary returned from `validation_step` (and similarly from `test_step` or `predict_step`) and save it to the specified CSV file. Each dictionary returned will be written as a row in the CSV file, with dictionary keys as column headers.

## Extending `TableWriter` (Advanced)

Similar to `FileWriter`, you can extend `TableWriter` by creating custom writer classes if you need more specialized table writing logic. Refer to the `lighter/callbacks/writer/table.py` file for details on how to create custom `TableWriter` classes.

## Recap: Persisting Experiment Outputs with Writers

Writers in Lighter are essential tools for saving and managing the outputs of your deep learning experiments. By using `FileWriter` and `TableWriter`, and by extending them with custom writer functions or classes when needed, you can:

*   Save model predictions in various formats for visualization and analysis.
*   Log metrics and tabular data to CSV files for experiment tracking and reporting.
*   Create highly customized output saving logic tailored to your specific research requirements.

With writers, Lighter provides a complete solution for not only running your deep learning experiments but also for effectively capturing and utilizing the valuable outputs they generate.

Next, explore the [How-To guide on Using Freezer](06_using_freezer.md) to learn how to freeze and unfreeze model layers during training, or return to the [How-To guides section](../how-to/) for more practical problem-solving guides. You can also go back to the [Explanation section](../explanation/) for more conceptual documentation or the [Tutorials section](../tutorials/) for end-to-end examples.
