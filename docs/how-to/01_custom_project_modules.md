# How to Use Custom Project Modules in Lighter

## Introduction to Custom Modules

Lighter is designed to be highly extensible, allowing you to seamlessly integrate your own custom modules (e.g., models, datasets, callbacks, metrics) into your deep learning projects. This how-to guide will walk you through the process of using custom modules in Lighter, enabling you to tailor the framework to your specific research or application needs.

Custom modules are Python modules that you create and maintain within your project directory, as opposed to relying solely on built-in Lighter modules or external libraries. Using custom modules offers several advantages:

*   **Encapsulation of Project-Specific Logic**: You can encapsulate your project's unique models, datasets, and other components within dedicated modules, keeping your project code organized and maintainable.
*   **Code Reusability**: Custom modules can be reused across different experiments and configurations within your project, promoting consistency and reducing code duplication.
*   **Version Control**: By keeping custom modules within your project, you can easily track changes, collaborate with others, and maintain version control for your project-specific code.
*   **Flexibility and Extensibility**: Custom modules provide maximum flexibility to extend Lighter's functionality and adapt it to your evolving research directions.

## Project Structure for Custom Modules

To effectively use custom modules in Lighter, it's recommended to organize your project with a clear directory structure. A typical Lighter project with custom modules might look like this:

```
my_project/
├── models/              # Directory for custom model modules
│   └── my_model.py      # Example custom model definition
├── datasets/            # Directory for custom dataset modules
│   └── my_dataset.py    # Example custom dataset definition
├── callbacks/           # Directory for custom callback modules (optional)
├── metrics/             # Directory for custom metric modules (optional)
├── utils/               # Directory for utility modules (optional)
├── config.yaml          # Lighter configuration file
└── train.py             # (Optional) Entry point script to run training
```

In this structure:

*   **`my_project/`**: This is the root directory of your Lighter project. You can name it according to your project's purpose.
*   **`models/`**, **`datasets/`**, **`callbacks/`**, **`metrics/`**, **`utils/`**: These are subdirectories within your project where you can organize your custom Python modules based on their type (models, datasets, callbacks, metrics, utility functions, etc.). You can create or omit these directories as needed, depending on your project's complexity and the types of custom modules you are using.
*   **`my_model.py`**, **`my_dataset.py`**, etc.: These are example Python files where you define your custom classes, functions, and logic for models, datasets, etc. You can create multiple Python files within each subdirectory to further organize your custom modules.
*   **`config.yaml`**: This is your Lighter configuration file, where you will specify how to use your custom modules.
*   **`train.py`** (optional): This is an optional entry point script that you can create to run your Lighter training workflows. You can also run Lighter directly from the command line using `lighter fit config.yaml`.

## Defining Custom Modules

Within the module directories (e.g., `models/`, `datasets/`), you define your custom Python modules as regular Python files (`.py`). Each file can contain one or more classes, functions, and variables, depending on your needs.

**Example: Custom Model (`my_project/models/my_model.py`)**

```python title="my_project/models/my_model.py"
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)
```

**Example: Custom Dataset (`my_project/datasets/my_dataset.py`)**

```python title="my_project/datasets/my_dataset.py"
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        # ... load data from data_path ...
        self.samples = [...] # List of data samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # ... preprocess sample ...
        if self.transform:
            sample = self.transform(sample)
        return sample
```

These are just simple examples. Your custom modules can be as complex as needed for your project.

## Importing Custom Modules in `config.yaml`

To use your custom modules in Lighter, you need to reference them in your `config.yaml` file using the `_target_` key. Lighter's dynamic module loading mechanism, powered by the `import_module_from_path` function, will then import and instantiate your custom modules at runtime.

### Specifying the `project` Path

First, you need to tell Lighter where to find your project directory by specifying the `project` key at the top level of your `config.yaml` file. The value of the `project` key should be the path to your project's root directory, relative to the location of the `config.yaml` file itself.

**Example: Specifying `project` Path in `config.yaml`**

```yaml title="config.yaml"
project: my_project/ # Path to the project root directory
```

### Referencing Custom Modules using `_target_`

Once you have specified the `project` path, you can reference your custom modules in the configuration using the `_target_` key, followed by the Python module path to your custom module, relative to the project root directory.

**Example: Using Custom Model and Dataset in `config.yaml`**

```yaml title="config.yaml"
project: my_project/ # Project root directory is 'my_project/'

system:
  model:
    _target_: my_project.models.MyModel # Load custom model from 'my_project/models/my_model.py'
    input_size: 784
    num_classes: 10

  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: my_project.datasets.MyDataset # Load custom dataset from 'my_project/datasets/my_dataset.py'
        data_path: "data/train.csv"
        # ... dataset arguments ...
      batch_size: 32
      shuffle: True
```

In this example:

*   `_target_: my_project.models.MyModel` tells Lighter to load the `MyModel` class from the `my_model.py` file located in the `models/` subdirectory within your project directory (`my_project/models/my_model.py`).
*   `_target_: my_project.datasets.MyDataset` tells Lighter to load the `MyDataset` class from the `my_dataset.py` file located in the `datasets/` subdirectory within your project directory (`my_project/datasets/my_dataset.py`).

Lighter will automatically prepend the `project` path (`my_project/`) to the module paths specified in `_target_` and use the `import_module_from_path` function to dynamically import your custom modules.

## Running Lighter with Custom Modules

To run your Lighter experiments that use custom modules, you simply execute the `lighter fit` command (or other Lighter CLI commands) with your `config.yaml` file, just as you would with built-in modules.

**Example: Running Training with Custom Modules**

```bash title="Terminal"
lighter fit --config config.yaml
```

As long as your `config.yaml` file correctly specifies the `project` path and the `_target_` paths to your custom modules, Lighter will dynamically load and use them during the experiment execution.

## Recap: Integrating Custom Modules into Lighter

Using custom project modules in Lighter is a straightforward process that involves:

1.  **Organizing your project** with a clear directory structure, including subdirectories for custom modules (e.g., `models/`, `datasets/`).
2.  **Defining your custom modules** (models, datasets, etc.) as Python files within your project directories.
3.  **Specifying the `project` path** in your `config.yaml` file.
4.  **Referencing your custom modules** in the `config.yaml` using the `_target_` key with module paths relative to your project root.
5.  **Running Lighter** with your `config.yaml` file as usual.

By following these steps, you can seamlessly integrate your project-specific code into Lighter workflows, leveraging the framework's flexibility and extensibility to build powerful and customized deep learning systems.

Next, explore the [How-To guide on Debugging Configuration Errors](02_debugging_config_errors.md) to learn how to troubleshoot common configuration issues, or return to the [How-To guides section](../how-to/) for more practical problem-solving guides. You can also go back to the [Explanation section](../explanation/) for more conceptual documentation or the [Tutorials section](../tutorials/) for end-to-end examples.
