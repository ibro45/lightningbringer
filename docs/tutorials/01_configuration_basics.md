# Configuration Basics

## Introduction

In machine learning experiments, managing configurations is crucial for reproducibility and flexibility. Configuration files allow you to define all aspects of your experiment in a declarative manner, from hyperparameters to dataset paths and model architectures. Lighter embraces this philosophy, making configuration the central piece of your workflow. This tutorial will guide you through the fundamentals of Lighter's configuration system, enabling you to effectively customize and manage your experiments.

## Basic Structure of a `config.yaml`

Lighter uses YAML files to define experiment configurations. A typical `config.yaml` file is organized into several key sections:

*   **`trainer`**: Configures the PyTorch Lightning `Trainer`, controlling aspects like accelerators, devices, and callbacks.
*   **`system`**: Defines the core components of your system, including the model, criterion, optimizer, scheduler, and dataloaders.
*   **`args`**: Allows you to override configuration values from the command line, providing flexibility for different runs.

Here's a minimal example illustrating the basic structure:

```yaml title="config.yaml"
trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: 10

system:
    _target_: lighter.System

    model:
        _target_: torch.nn.Linear
        in_features: 100
        out_features: 10

    criterion:
        _target_: torch.nn.CrossEntropyLoss

    optimizer:
        _target_: torch.optim.Adam
        params: "$@system#model.parameters()"
        lr: 0.001

    dataloaders:
        train:
            _target_: torch.utils.data.DataLoader
            dataset:
                _target_: torch.utils.data.TensorDataset
                tensors:
                    - _target_: torch.randn
                      size: [1000, 100]
                    - _target_: torch.randint
                      low: 0
                      high: 10
                      size: [1000]
            batch_size: 32
```

In this example, we define a simple linear model, a cross-entropy loss, and an Adam optimizer. The `dataloaders` section sets up a basic training dataloader using random tensors.

### Stages in Lighter

Lighter operates with the concept of "stages" - `fit`, `validate`, `test`, and `predict`.  When you run a command like `lighter fit`, Lighter prunes the configuration to only include the relevant sections for the `fit` stage. This ensures that only necessary components are initialized for each stage, optimizing resource usage and clarity.

### Command Line Arguments with `args`

The `args` section in the `config.yaml` provides a powerful mechanism to override configuration values directly from the command line. This is particularly useful for hyperparameter tuning or running quick experiments with different settings without modifying the base configuration file. We will explore this in more detail later in this tutorial.

## MONAI Bundle Syntax for Class Instantiation

Lighter adopts the MONAI Bundle configuration style, which offers a concise and flexible way to specify classes and their arguments within the YAML configuration. The key element of this syntax is the `_target_` key.

To instantiate any class, you specify its fully qualified name using the `_target_` key, and its constructor arguments as key-value pairs under the same section.

For example, to define a `torch.optim.Adam` optimizer:

```yaml
optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
    weight_decay: 1.0e-5
```

Here, `_target_: torch.optim.Adam` tells Lighter to instantiate the `Adam` class from the `torch.optim` module.  `lr: 0.001` and `weight_decay: 1.0e-5` are passed as arguments to the `Adam` constructor.

This syntax extends to all configurable components in Lighter, including models, datasets, transforms, metrics, callbacks, and more.

### Example: Defining a Model

```yaml
system:
  model:
    _target_: torchvision.models.resnet18
    pretrained: false
    num_classes: 10
```

This configuration snippet defines a `resnet18` model from `torchvision.models`, setting `pretrained` to `false` and `num_classes` to `10`.

### Example: Defining a Dataset

```yaml
system:
  dataloaders:
    train:
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: .datasets/
        download: true
        transform:
          _target_: torchvision.transforms.ToTensor
```

This defines the CIFAR10 dataset with a `ToTensor` transform.

## Validation with Cerberus

Lighter utilizes [Cerberus](https:// Cerberus.readthedocs.io/en/stable/) for configuration validation. Cerberus is a powerful validation library that ensures your `config.yaml` adheres to a predefined schema. This schema, defined within Lighter, specifies the expected structure and data types for each configuration section.

When you run Lighter, the configuration file is automatically validated against this schema. If any discrepancies are found, Lighter will raise a `ValidationError`, providing informative messages about the invalid parts of your configuration.

### Example Validation Error

Let's say you incorrectly specify `max_epochs` as a string instead of an integer in your `config.yaml`:

```yaml
trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: "string"  # Incorrect type - should be an integer
```

Running Lighter with this configuration will result in a `ValidationError` similar to:

```
ValidationError: Configuration validation failed.
Path: 'trainer.max_epochs', Error: Must be of integer type
```

These validation errors are designed to help you quickly identify and fix issues in your configuration, ensuring that your experiments are set up correctly.

## Overriding Configuration from the CLI

Lighter's `args` section in the `config.yaml` allows you to override configuration values directly from the command line. This is incredibly useful for tasks like hyperparameter tuning, running ablation studies, or quickly adjusting settings without modifying your base configuration file.

The `args` section is structured to mirror the main configuration, allowing you to target specific parameters for overriding.

### Basic Overriding Syntax

To override a value, you use dot notation to specify the path to the parameter you want to change. For example, to override `trainer.max_epochs` to `20` from the command line, you would use:

```bash
lighter fit config.yaml args.fit.trainer.max_epochs=20
```

Here, `args.fit.trainer.max_epochs=20` is appended to the `lighter fit config.yaml` command.  `fit` specifies that this override applies to the `fit` stage.

### Overriding Nested Parameters

You can override nested parameters in a similar fashion. For instance, to change the learning rate of the optimizer:

```bash
lighter fit config.yaml args.fit.system.optimizer.lr=0.01
```

### Overriding Multiple Parameters

You can override multiple parameters in a single command by separating the overrides with spaces:

```bash
lighter fit config.yaml args.fit.trainer.max_epochs=20 args.fit.system.optimizer.lr=0.01
```

### Structure of the `args` Section

The `args` section in your `config.yaml` should be structured to reflect the stages and sections you want to override. For example:

```yaml title="config.yaml"
trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: 10

system:
    _target_: lighter.System
    optimizer:
        _target_: torch.optim.Adam
        lr: 0.001

args:
    fit: # Overrides for the 'fit' stage
        trainer:
            max_epochs: 20 # Example override for max_epochs
        system:
            optimizer:
                lr: 0.01 # Example override for learning rate
```

In this `config.yaml`, the `args.fit` section pre-defines potential overrides for the `fit` stage. However, these overrides are only applied if you explicitly specify them in the command line. If you run `lighter fit config.yaml` without any `args` overrides, the values defined in the main `trainer` and `system` sections will be used (e.g., `max_epochs` will be `10`, and `lr` will be `0.001`).

## Advanced Configuration Customization

Lighter's configuration system supports advanced customization options, including nested configurations and references within the config file.

### Nested Configurations

You can create nested configurations to organize your `config.yaml` more effectively. This is particularly useful for complex experiments with many parameters.

Example of nested configuration:

```yaml
system:
  model:
    _target_: my_project.models.ComplexModel
    encoder:
      _target_: my_project.models.Encoder
      num_layers: 4
      hidden_dim: 256
    decoder:
      _target_: my_project.models.Decoder
      num_layers: 2
      output_dim: 10
```

Here, the `model` section has nested `encoder` and `decoder` configurations, each defining its own class and arguments.

### References within the Config File

Lighter allows you to create references within your `config.yaml` to avoid repetition and improve maintainability. You can reference any parameter defined elsewhere in the configuration file using the syntax `"$@[section]#[parameter_path]"`.

In the basic structure example, we saw this in action:

```yaml
optimizer:
    _target_: torch.optim.Adam
    params: "$@system#model.parameters()" # Reference to model parameters
    lr: 0.001
```

`params: "$@system#model.parameters()"` references the parameters of the `model` defined in the `system` section. This ensures that the optimizer is always configured to optimize the correct model parameters.

You can use references to link any part of your configuration, making it more dynamic and less prone to errors.

## Recap and Next Steps

In this tutorial, you've learned the fundamental aspects of Lighter's configuration system:

*   The basic structure of a `config.yaml` file and its key sections.
*   The MONAI Bundle syntax for instantiating classes using `_target_`.
*   How Lighter validates configurations using Cerberus and handles validation errors.
*   How to override configuration values from the command line using the `args` section.
*   Advanced customization options like nested configurations and references.

With this knowledge, you are well-equipped to create and customize configuration files for your deep learning experiments in Lighter.

In the next tutorials, we will explore more practical examples, such as [end-to-end image classification](02_image_classification.md) and [semantic segmentation](03_semantic_segmentation.md). You can also refer to the [How-To guides](../how-to/02_debugging_config_errors.md) for debugging common configuration issues and the [Explanation section](../explanation/02_configuration_system.md) for a deeper dive into the configuration system's design.
