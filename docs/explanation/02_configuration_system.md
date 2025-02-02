# Deep Dive into Lighter's Configuration System

## Introduction to Configuration in Lighter

Configuration is central to Lighter's design. It allows you to define and manage all aspects of your deep learning experiments in a declarative and organized manner, using human-readable YAML files. This approach offers numerous benefits:

*   **Experiment Reproducibility**: Configuration files explicitly capture all settings, ensuring experiments can be reliably reproduced.
*   **Simplified Experiment Management**: Manage complex experiments with a single `config.yaml` file, rather than scattered code.
*   **Parameter Sweeping**: Easily modify and iterate on experiments by changing configuration values.
*   **Collaboration**: Share and discuss experiments by sharing the configuration file.

This document provides a deep dive into Lighter's configuration system, explaining its structure, key features, and how to use it effectively.

## YAML Structure of `config.yaml`

Lighter configuration files are written in YAML (YAML Ain't Markup Language), a human-friendly data serialization format. A typical Lighter `config.yaml` file has a hierarchical structure with key-value pairs, organized into sections.

**Top-Level Keys**:

A Lighter configuration file typically includes the following top-level keys:

*   **`trainer`**: Configuration for the PyTorch Lightning `Trainer`. This section defines how the training process is executed, including settings for accelerators, devices, epochs, callbacks, loggers, and more.
*   **`system`**: Configuration for the `lighter.System` class, which encapsulates your deep learning system. This section defines:
    *   `model`: The neural network model.
    *   `dataloaders`: Data loaders for training, validation, testing, and prediction.
    *   `optimizer`: The optimizer algorithm.
    *   `scheduler`: The learning rate scheduler.
    *   `criterion`: The loss function (criterion).
    *   `metrics`: Evaluation metrics.
    *   `adapters`: Adapters for customizing data handling and argument passing.
    *   `inferer`: Inferer for controlling the inference process.
*   **`project`**: (Optional) Specifies the path to your project directory, enabling Lighter to dynamically load custom modules (models, datasets, etc.) from your project.
*   **`args`**: (Optional) Allows you to override configuration values from the command line when running Lighter.

**Example `config.yaml` Structure**:

```yaml title="config.yaml"
trainer: # PyTorch Lightning Trainer configuration
  _target_: pytorch_lightning.Trainer
  max_epochs: 100
  accelerator: "auto"
  devices: 1

system: # lighter.System configuration
  _target_: lighter.System

  model: # Model definition
    _target_: my_project.models.MyModel # Path to custom model class
    input_size: [1, 28, 28]
    num_classes: 10

  dataloaders: # DataLoaders configuration
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: my_project.datasets.MyDataset # Path to custom dataset class
        data_path: "data/train.csv"
      batch_size: 32
      shuffle: true
    val: # Validation dataloader
      _target_: torch.utils.data.DataLoader
      # ... (validation dataloader config) ...

  optimizer: # Optimizer configuration
    _target_: torch.optim.Adam
    lr: 1.0e-3
    weight_decay: 1.0e-5

  criterion: # Loss function configuration
    _target_: torch.nn.CrossEntropyLoss

  metrics: # Metrics configuration
    train:
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10
    val:
      - _target_: torchmetrics.Accuracy
        # ... (accuracy metric config) ...

project: my_project/ # Project path for dynamic module loading

args: # Command-line argument overrides
  fit: # 'lighter fit' stage arguments
    trainer:
      max_epochs: 20 # Override max_epochs to 20 when using 'lighter fit'
```

## MONAI Bundle Style Configuration

Lighter adopts the configuration style from MONAI Bundle, which is designed for modularity and flexibility in defining deep learning components. This style relies heavily on the `_target_` key to specify Python classes and their arguments.

**`_target_` Key**:

The `_target_` key is used throughout Lighter configurations to indicate the Python class that should be instantiated. The value of `_target_` is a string representing the Python path to the class (e.g., `torch.optim.Adam`, `my_project.models.MyModel`).

**Example: Defining a Model**:

```yaml
system:
  model:
    _target_: my_project.models.MyModel # Path to custom model class
    input_size: [1, 28, 28]           # Argument for MyModel.__init__
    num_classes: 10              # Argument for MyModel.__init__
```

In this example:

*   `_target_: my_project.models.MyModel` tells Lighter to instantiate the `MyModel` class from the `my_project.models` module.
*   `input_size: [1, 28, 28]` and `num_classes: 10` are passed as keyword arguments to the `MyModel` class's `__init__` constructor.

**Benefits of `_target_` Style**:

*   **Declarative Configuration**: You declare *what* you want to create (which class to use) and *how* to create it (constructor arguments) in the config file, rather than writing imperative code.
*   **Dynamic Instantiation**: Lighter dynamically instantiates classes based on the configuration, allowing for highly flexible and modular setups.
*   **Integration with Custom Modules**: Easily integrate your own custom models, datasets, metrics, and other components by specifying their Python paths in the config.

## Stage-Specific Configurations

Lighter uses the concept of **stages** to manage configurations for different phases of a deep learning workflow. The stages are:

*   **`fit`**: Training and validation stage (`lighter fit`).
*   **`validate`**: Validation stage (`lighter validate`).
*   **`test`**: Testing stage (`lighter test`).
*   **`predict`**: Prediction stage (`lighter predict`).

**Configuration Pruning**:

When you run Lighter for a specific stage (e.g., `lighter fit`), Lighter **prunes** the configuration file, keeping only the relevant sections for that stage. This ensures that only necessary components are initialized and used, optimizing resource usage and avoiding conflicts.

**Example: Stage-Specific DataLoaders**:

```yaml
system:
  dataloaders:
    train: # Dataloader for 'fit' stage
      _target_: torch.utils.data.DataLoader
      # ... (train dataloader config) ...
    val: # Dataloader for 'fit' and 'validate' stages
      _target_: torch.utils.data.DataLoader
      # ... (validation dataloader config) ...
    test: # Dataloader for 'test' stage
      _target_: torch.utils.data.DataLoader
      # ... (test dataloader config) ...
    predict: # Dataloader for 'predict' stage
      _target_: torch.utils.data.DataLoader
      # ... (predict dataloader config) ...
```

In this example, different dataloaders are defined for `train`, `val`, `test`, and `predict` stages. When you run `lighter fit`, Lighter will use the `train` and `val` dataloaders. When you run `lighter validate`, it will use the `val` dataloader. Similarly, `lighter test` will use `test`, and `lighter predict` will use `predict`.

**`Resolver` Class**:

Lighter uses a `Resolver` class (in `lighter/engine/resolver.py`) to handle configuration loading, parsing, and stage-specific pruning. The `Resolver` is responsible for:

1.  **Loading the `config.yaml` file**.
2.  **Validating the configuration** against a predefined schema (using Cerberus, see below).
3.  **Resolving `_target_` paths** and dynamically importing modules.
4.  **Pruning the configuration** based on the current stage.
5.  **Instantiating configured objects** (models, datasets, etc.).

## Validation with Cerberus

Lighter uses **Cerberus**, a powerful validation library for Python, to ensure that your `config.yaml` files adhere to a predefined schema. The schema defines the expected structure, data types, and required fields for the configuration.

**`SCHEMA` in `engine/config.py`**:

The configuration schema is defined in the `SCHEMA` dictionary within the `lighter/engine/config.py` file. This schema specifies the valid structure and rules for Lighter configurations.

**Example Schema Snippet (from `engine/config.py`)**:

```python
SCHEMA = {
    "trainer": { # Schema for 'trainer' section
        "type": "dict",
        "required": False,
        "schema": {
            "_target_": {"type": "string", "required": True}, # '_target_' is required
            "max_epochs": {"type": "integer", "required": False, "min": 1}, # 'max_epochs' is optional, must be integer >= 1
            "accelerator": {"type": "string", "required": False}, # 'accelerator' is optional, must be string
            # ... (other trainer schema rules) ...
        },
    },
    "system": { # Schema for 'system' section
        "type": "dict",
        "required": True, # 'system' section is required
        "schema": {
            "_target_": {"type": "string", "required": True}, # '_target_' is required
            "model": {"type": "dict", "required": True}, # 'model' is required, must be dict
            "dataloaders": {"type": "dict", "required": True}, # 'dataloaders' is required, must be dict
            "optimizer": {"type": "dict", "required": True}, # 'optimizer' is required, must be dict
            "scheduler": {"type": "dict", "required": False}, # 'scheduler' is optional
            "criterion": {"type": "dict", "required": True}, # 'criterion' is required
            "metrics": {"type": "dict", "required": False}, # 'metrics' is optional
            "adapters": {"type": "dict", "required": False}, # 'adapters' is optional
            "inferer": {"type": "dict", "required": False}, # 'inferer' is optional
            # ... (other system schema rules) ...
        },
    },
    # ... (schema for other top-level keys) ...
}
```

**Validation Process**:

When Lighter loads your `config.yaml` file, it automatically validates it against the `SCHEMA`. If the configuration is invalid (e.g., missing required fields, incorrect data types), Lighter will raise a `ConfigurationError` with detailed error messages, helping you quickly identify and fix issues in your config file.

**Benefits of Validation**:

*   **Early Error Detection**: Catch configuration errors early in the experiment setup process, before runtime.
*   **Configuration Consistency**: Ensure that your configuration files follow a consistent structure and adhere to predefined rules.
*   **Improved User Experience**: Provide helpful error messages to guide users in creating valid configuration files.

## Overriding Configuration from the CLI

Lighter allows you to override configuration values directly from the command line when running Lighter CLI commands (e.g., `lighter fit`, `lighter validate`). This is achieved using the `args` section in your `config.yaml` file.

**`args` Section**:

The `args` section is an optional top-level key in `config.yaml`. It allows you to define stage-specific argument overrides.

**Example `args` Section**:

```yaml title="config.yaml"
args:
  fit: # Arguments for 'lighter fit' stage
    trainer: # Override 'trainer' section
      max_epochs: 20 # Override 'trainer.max_epochs' to 20
      accelerator: "gpu" # Override 'trainer.accelerator' to "gpu"
    system: # Override 'system' section
      optimizer:
        lr: 1.0e-4 # Override 'system.optimizer.lr' to 1.0e-4
  validate: # Arguments for 'lighter validate' stage
    trainer:
      devices: 2 # Override 'trainer.devices' to 2 for validation
```

**Command-Line Overrides**:

When you run a Lighter command with a `config.yaml` file that includes the `args` section, Lighter will apply the specified overrides.

**Example Usage**:

```bash title="Terminal"
lighter fit --config config.yaml # Runs 'fit' stage with overrides defined under 'args.fit'
lighter validate --config config.yaml # Runs 'validate' stage with overrides under 'args.validate'
```

In the `config.yaml` example above:

*   When you run `lighter fit`, the `trainer.max_epochs` will be overridden to `20`, `trainer.accelerator` to `"gpu"`, and `system.optimizer.lr` to `1.0e-4`.
*   When you run `lighter validate`, the `trainer.devices` will be overridden to `2`.

**Order of Precedence**:

Command-line argument overrides have the **highest precedence**. If you specify a value in the `args` section and also pass a command-line argument that modifies the same value, the command-line argument will take effect.

**Benefits of CLI Overrides**:

*   **Experiment Flexibility**: Easily modify experiment settings without editing the `config.yaml` file directly.
*   **Parameter Tuning**: Quickly run parameter sweeps and ablation studies by overriding specific parameters from the command line.
*   **Stage-Specific Adjustments**: Apply different settings for different stages (e.g., use more devices for validation than training).

## Advanced Configuration Features

Lighter's configuration system offers several advanced features for more complex and flexible experiment setups:

*   **Nested Configurations**: You can create nested dictionaries and lists within your `config.yaml` to organize complex configurations hierarchically.
*   **References**: You can use references within the config file to reuse values and avoid repetition. (Further documentation on references will be added in future updates).
*   **Environment Variables**: (Planned) Support for using environment variables within configurations for sensitive information or dynamic settings.
*   **Custom Resolvers**: (Planned) Ability to define custom resolvers for more advanced configuration logic.

## Recap: Mastering Lighter's Configuration

Lighter's configuration system is a powerful tool for streamlining your deep learning experiments. By understanding its YAML structure, `_target_` style, stage-specific configurations, validation with Cerberus, and CLI overrides, you can effectively manage complex experiments, enhance reproducibility, and focus on your research goals.

Next, explore the [Adapter System Explanation](../explanation/03_adapter_system.md) to understand how adapters provide further customization in Lighter, or return to the [Explanation section](../explanation/) for more conceptual documentation. You can also refer back to the [How-To guides section](../how-to/) for practical problem-solving guides or the [Tutorials section](../tutorials/) for end-to-end examples.
