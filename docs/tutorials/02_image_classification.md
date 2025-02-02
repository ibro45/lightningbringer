# End-to-End Image Classification

## Introduction

Image classification is a fundamental task in computer vision, with applications ranging from image tagging to medical diagnosis. In this tutorial, we will guide you through the process of training an image classification model using Lighter on a standard dataset, CIFAR10. We will cover dataset loading, model definition, configuration setup, training execution, and evaluation.

## Dataset: CIFAR10

We will use the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), a widely used dataset for image classification. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The dataset is conveniently available through torchvision.

### Loading CIFAR10 with Lighter

To use CIFAR10 in Lighter, you need to configure the `dataloaders` section in your `config.yaml`. Here's how you can define the training dataloader:

```yaml title="config.yaml"
system:
  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: .datasets/
        download: true
        train: true
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.5, 0.5, 0.5]
              std: [0.5, 0.5, 0.5]
      batch_size: 32
      shuffle: true
      num_workers: 4 # Adjust based on your system
```

Let's break down this configuration:

*   **`_target_: torch.utils.data.DataLoader`**:  Specifies that we are using PyTorch's `DataLoader` for loading the dataset.
*   **`dataset`**: Defines the dataset itself.
    *   **`_target_: torchvision.datasets.CIFAR10`**:  Indicates we are using the CIFAR10 dataset from torchvision.
    *   **`root: .datasets/`**:  Specifies the directory where the dataset will be downloaded or loaded from.
    *   **`download: true`**:  If the dataset is not found at `root`, it will be downloaded.
    *   **`train: true`**:  Indicates that we are loading the training set.
    *   **`transform`**: Defines the data transformations to be applied to the images.
        *   **`_target_: torchvision.transforms.Compose`**:  Used to chain multiple transforms together.
        *   **`transforms`**: A list of transforms.
            *   **`_target_: torchvision.transforms.ToTensor`**: Converts PIL images to PyTorch tensors.
            *   **`_target_: torchvision.transforms.Normalize`**: Normalizes the image tensors with the specified mean and standard deviation.
*   **`batch_size: 32`**:  Sets the batch size for training.
*   **`shuffle: true`**:  Shuffles the training data at the beginning of each epoch.
*   **`num_workers: 4`**:  Specifies the number of worker processes for data loading (adjust this based on your CPU and system resources).

You can similarly define a validation dataloader if needed, by setting `train: false` in the dataset configuration and potentially using a different set of transforms.

## Model: Simple CNN

For this tutorial, we will use a simple Convolutional Neural Network (CNN) for image classification. You can define this model in a Python file (e.g., `my_project/models/simple_cnn.py`) and then import it into your configuration.

```python title="my_project/models/simple_cnn.py"
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 8 * 8, num_classes) # Assuming 32x32 images

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
```

This simple CNN consists of two convolutional layers with ReLU activations and max-pooling, followed by a flattening layer and a fully connected layer for classification.

### Defining the Model in `config.yaml`

To use this model in Lighter, you need to specify it in the `system.model` section of your `config.yaml`. Assuming your project structure is set up as described in the [Custom Project Modules How-To guide](../how-to/custom_project_modules.md), you can configure the model as follows:

```yaml title="config.yaml"
system:
  model:
    _target_: my_project.models.simple_cnn.SimpleCNN
    num_classes: 10 # Matches CIFAR10 classes
```

Here, `_target_: my_project.models.simple_cnn.SimpleCNN` tells Lighter to load the `SimpleCNN` class from the `my_project.models.simple_cnn` module. `num_classes: 10` is passed as an argument to the `SimpleCNN` constructor.

## Complete Configuration (`config.yaml`)

Now, let's put together the complete `config.yaml` file for training the SimpleCNN on CIFAR10:

```yaml title="config.yaml"
trainer:
  accelerator: "auto" # Use GPU if available, else CPU
  max_epochs: 10

system:
  _target_: lighter.System

  model:
    _target_: my_project.models.simple_cnn.SimpleCNN
    num_classes: 10

  criterion:
    _target_: torch.nn.CrossEntropyLoss

  optimizer:
    _target_: torch.optim.Adam
    lr: 1.0e-3

  metrics:
    train:
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10
    val:
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10
    test:
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10

  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: .datasets/
        download: true
        train: true
        transform:
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.5, 0.5, 0.5]
              std: [0.5, 0.5, 0.5]
      batch_size: 32
      shuffle: true
      num_workers: 4
    val: # Optional validation dataloader
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: .datasets/
        download: true
        train: false # Load validation set (test set of CIFAR10)
        transform: # Use the same transforms as training, or define different ones
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.5, 0.5, 0.5]
              std: [0.5, 0.5, 0.5]
      batch_size: 32
      num_workers: 4
    test: # Optional test dataloader (you can also use 'lighter validate' on val dataloader)
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: .datasets/
        download: true
        train: false # Load test set (test set of CIFAR10)
        transform: # Use the same transforms as training, or define different ones
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.5, 0.5, 0.5]
              std: [0.5, 0.5, 0.5]
      batch_size: 32
      num_workers: 4
```

This configuration defines all the necessary components for training, validation, and testing:

*   **`trainer`**: Configures the PyTorch Lightning Trainer to use automatic accelerator selection and train for a maximum of 10 epochs.
*   **`system`**: Defines the Lighter System.
    *   **`model`**: Specifies the `SimpleCNN` model.
    *   **`criterion`**: Sets the loss function to `CrossEntropyLoss`.
    *   **`optimizer`**: Uses the `Adam` optimizer with a learning rate of 1.0e-3.
    *   **`metrics`**: Defines accuracy metrics for training, validation, and testing stages.
    *   **`dataloaders`**: Configures `DataLoader`s for `train`, `val`, and `test` stages, using the CIFAR10 dataset and appropriate transforms.

## Training Execution

To start training, save the above configuration as `config.yaml` in your project directory. Ensure that you have created the `my_project/models/simple_cnn.py` file as well. Then, open your terminal, navigate to your project directory, and run the following command:

```bash title="Terminal"
lighter fit --config config.yaml
```

Lighter will parse your `config.yaml`, initialize all the components, and start the training process using PyTorch Lightning. You will see the training progress, including loss and metrics, logged in your terminal.

## Evaluation

After training, you can evaluate your model on the validation or test set. To run validation, use:

```bash title="Terminal"
lighter validate --config config.yaml
```

Or, to run testing:

```bash title="Terminal"
lighter test --config config.yaml
```

Lighter will load the best checkpoint saved during training (if a `ModelCheckpoint` callback is used in the configuration, which is often the default in more complex setups) and evaluate the model on the specified dataloader, reporting the metrics defined in the `system.metrics` section for the `val` or `test` stage, respectively.

## Recap and Next Steps

In this tutorial, you have successfully trained and evaluated an image classification model on the CIFAR10 dataset using Lighter. You learned how to:

*   Configure dataloaders for image datasets using torchvision.
*   Define a simple CNN model and integrate it into your Lighter configuration.
*   Create a complete `config.yaml` file for an image classification experiment.
*   Execute training and evaluation using the Lighter CLI.

This tutorial provides a solid foundation for building more complex image classification experiments with Lighter. In the next tutorials, we will explore [semantic segmentation](03_semantic_segmentation.md) and [transfer learning](04_transfer_learning.md). You can also refer to the [How-To guides](../how-to/01_custom_project_modules.md) for using custom modules and the [Explanation section](../explanation/01_overview.md) for a deeper understanding of Lighter's design principles.
