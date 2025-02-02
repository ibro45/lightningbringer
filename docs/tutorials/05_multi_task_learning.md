# Multi-Task Learning: Training a Model for Multiple Tasks

## Introduction

Multi-task learning is a subfield of machine learning where a single model is trained to perform multiple related tasks simultaneously. This approach can lead to improved generalization and efficiency compared to training separate models for each task. Multi-task learning is particularly beneficial when tasks share common features or when data is limited for individual tasks but abundant across tasks.

In this tutorial, we will demonstrate how to train a multi-task model using Lighter. We will consider a simplified multi-task scenario with two tasks: image classification and image segmentation. We will use a shared ResNet-18 backbone with separate heads for each task.

## Tasks: Image Classification and Semantic Segmentation

We will define two related tasks:

1.  **Image Classification**: Classifying an input image into one of 10 classes (CIFAR-10 classes).
2.  **Semantic Segmentation**: Segmenting objects in the same input image into two classes (e.g., object vs. background). For simplicity, we will create synthetic segmentation labels.

We will use the CIFAR-10 dataset for both tasks. For segmentation labels, we will generate synthetic binary masks where objects (e.g., airplanes, cars, etc.) are considered the foreground, and the background is the background.

## Dataset: CIFAR-10 for Multi-Task Learning

We will reuse the CIFAR-10 dataset and augment it with synthetic segmentation labels. We will create a custom dataset class that returns both classification labels and segmentation masks.

### Custom Multi-Task CIFAR-10 Dataset

Create a new Python file, e.g., `projects/cifar10/datasets/multitask_cifar10.py`, and define the custom dataset:

```python title="projects/cifar10/datasets/multitask_cifar10.py"
import torch
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

class MultiTaskCIFAR10(CIFAR10):
    def __getitem__(self, index):
        image, label = super().__getitem__(index) # Original CIFAR-10 data

        # Generate synthetic segmentation mask (simple example: binary mask)
        mask = torch.zeros(32, 32, dtype=torch.long)
        if label < 5: # First 5 classes as "object"
            mask[10:20, 10:20] = 1 # Create a square object in the center

        return image, {"classification": label, "segmentation": mask}

def build_multitask_cifar10_dataset(
    root: str = ".datasets/",
    train: bool = True,
    download: bool = False,
    transform = None,
    **kwargs,
):
    return MultiTaskCIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform,
        **kwargs,
    )
```

This custom dataset `MultiTaskCIFAR10` inherits from `CIFAR10` and overrides the `__getitem__` method to return a dictionary containing both `classification` labels (original CIFAR-10 labels) and synthetic `segmentation` masks. The `build_multitask_cifar10_dataset` function is a helper function to instantiate the dataset, which we will use in the config.

### Configuring Multi-Task Dataset in `config.yaml`

Configure the multi-task CIFAR-10 dataset in the `system.dataloaders` section of your `config.yaml`:

```yaml title="config.yaml"
project: projects/cifar10/ # Project path

system:
  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: datasets.multitask_cifar10.build_multitask_cifar10_dataset # Custom dataset
        root: ".datasets/"
        train: true
        download: true
        transform: # Transforms for both tasks (can be task-specific if needed)
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.RandomResizedCrop
              size: 224
              scale: [0.8, 1.0]
              ratio: [0.75, 1.3333]
            - _target_: torchvision.transforms.RandomHorizontalFlip
              p: 0.5
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
      batch_size: 32
      shuffle: true
      num_workers: 4
    val: # Validation dataloader
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: datasets.multitask_cifar10.build_multitask_cifar10_dataset # Custom dataset
        root: ".datasets/"
        train: false
        download: false
        transform: # Validation transforms
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.Resize
              size: 256
            - _target_: torchvision.transforms.CenterCrop
              size: 224
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
      batch_size: 32
      num_workers: 4
```

*   **`project: projects/cifar10/`**:  Specifies the project path so Lighter can dynamically import modules from `projects/cifar10/`.
*   **`dataset._target_: datasets.multitask_cifar10.build_multitask_cifar10_dataset`**:  Uses the custom `build_multitask_cifar10_dataset` function we defined.

## Model: Shared Backbone with Task-Specific Heads

We will use a ResNet-18 model as a shared backbone and add separate heads for classification and segmentation.

### Custom Multi-Task Model

Create a new Python file, e.g., `projects/cifar10/models/multitask_net.py`, and define the multi-task model:

```python title="projects/cifar10/models/multitask_net.py"
import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskNet(nn.Module):
    def __init__(self, num_classes_classification=10, num_classes_segmentation=2):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True) # Shared backbone (pretrained ResNet-18)
        self.backbone.fc = nn.Identity() # Remove original FC layer

        # Classification head
        self.classification_head = nn.Linear(self.backbone.fc.in_features, num_classes_classification)
        # Segmentation head (simplified example: linear layer for per-pixel classification)
        self.segmentation_head = nn.Conv2d(self.backbone.fc.in_features, num_classes_segmentation, kernel_size=1) # 1x1 conv for segmentation

    def forward(self, x):
        features = self.backbone(x) # Shared backbone features

        # Classification task
        classification_logits = self.classification_head(features)

        # Segmentation task
        segmentation_logits = self.segmentation_head(features.unsqueeze(-1).unsqueeze(-1)) # Project features to 2D and apply 1x1 conv

        return {"classification": classification_logits, "segmentation": segmentation_logits}
```

This `MultiTaskNet` model:

*   Uses a pre-trained ResNet-18 as a shared backbone (`self.backbone`).
*   Removes the original fully connected layer of ResNet-18.
*   Adds a `classification_head` (linear layer) for the classification task.
*   Adds a `segmentation_head` (1x1 convolutional layer) for the segmentation task.
*   The `forward` method returns a dictionary containing logits for both tasks.

### Configuring Multi-Task Model in `config.yaml`

Configure the multi-task model in the `system.model` section:

```yaml title="config.yaml"
system:
  model:
    _target_: models.multitask_net.MultiTaskNet # Custom multi-task model
    num_classes_classification: 10 # CIFAR-10 classes
    num_classes_segmentation: 2 # Segmentation classes (object vs. background)
```

*   **`model._target_: models.multitask_net.MultiTaskNet`**: Uses the custom `MultiTaskNet` model.
*   **`num_classes_classification` and `num_classes_segmentation`**:  Pass task-specific parameters to the model.

## Multi-Task Criterion (Loss Function)

We need to define separate loss functions for each task and combine them into a multi-task loss.

### Custom MultiTaskLoss

Create a new Python file, e.g., `projects/cifar10/losses/multitask_loss.py`, and define the `MultiTaskLoss`:

```python title="projects/cifar10/losses/multitask_loss.py"
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, losses, weights):
        super().__init__()
        self.losses = nn.ModuleDict(losses) # Dictionary of loss functions
        self.weights = weights # Dictionary of weights for each loss

    def forward(self, outputs, targets):
        total_loss = 0
        loss_dict = {}
        for task_name, loss_fn in self.losses.items():
            task_output = outputs[task_name]
            task_target = targets[task_name]
            task_loss = loss_fn(task_output, task_target)
            total_loss += task_loss * self.weights.get(task_name, 1.0) # Apply weight, default to 1.0
            loss_dict[task_name] = task_loss

        loss_dict["total"] = total_loss # Add total loss
        return loss_dict
```

This `MultiTaskLoss` module:

*   Takes a dictionary of loss functions (`losses`) and weights (`weights`) as input.
*   Calculates individual losses for each task.
*   Combines the losses into a `total_loss` using the specified weights.
*   Returns a dictionary containing individual task losses and the `total_loss`.

### Configuring Multi-Task Criterion in `config.yaml`

Configure the `MultiTaskLoss` in the `system.criterion` section:

```yaml title="config.yaml"
system:
  criterion:
    _target_: losses.multitask_loss.MultiTaskLoss # Custom multi-task loss
    losses: # Dictionary of loss functions for each task
      classification:
        _target_: torch.nn.CrossEntropyLoss
      segmentation:
        _target_: torch.nn.CrossEntropyLoss # Example: CrossEntropyLoss for segmentation
    weights: # Weights for each loss (optional)
      classification: 1.0
      segmentation: 0.5 # Lower weight for segmentation (synthetic labels)
```

*   **`criterion._target_: losses.multitask_loss.MultiTaskLoss`**: Uses the custom `MultiTaskLoss`.
*   **`losses`**: Defines a dictionary of loss functions, one for each task. We use `CrossEntropyLoss` for both classification and segmentation in this example.
*   **`weights`**:  Optionally, you can specify weights for each loss to balance the contribution of different tasks to the total loss. Here, we give segmentation loss a lower weight (0.5) as the labels are synthetic.

## Metrics

We will use accuracy for the classification task and Dice metric for the segmentation task.

### Configuring Multi-Task Metrics

Configure the metrics in the `system.metrics` section:

```yaml title="config.yaml"
system:
  metrics:
    train: # Metrics for training
      classification: # Metrics for classification task
        - _target_: torchmetrics.Accuracy
          task: "multiclass"
          num_classes: 10
      segmentation: # Metrics for segmentation task
        - _target_: torchmetrics.DiceMetric
          task: "binary" # Binary segmentation
    val: # Metrics for validation
      classification:
        - _target_: torchmetrics.Accuracy
          task: "multiclass"
          num_classes: 10
      segmentation:
        - _target_: torchmetrics.DiceMetric
          task: "binary"
    test: # Metrics for test
      classification:
        - _target_: torchmetrics.Accuracy
          task: "multiclass"
          num_classes: 10
      segmentation:
        - _target_: torchmetrics.DiceMetric
          task: "binary"
```

We define a nested dictionary structure for metrics:

*   The outer keys (`train`, `val`, `test`) represent the stage.
*   The inner keys (`classification`, `segmentation`) represent the task.
*   We use `torchmetrics.Accuracy` for classification and `torchmetrics.DiceMetric` for segmentation.

## Adapters for Multi-Task Learning

We need to define adapters to handle the multi-task data format and loss/metric arguments.

### Configuring Adapters

Configure the adapters in the `system.adapters` section:

```yaml title="config.yaml"
system:
  adapters:
    train: # Adapters for training stage
      batch: # Batch adapter
        _target_: lighter.adapters.BatchAdapter
        input_accessor: 0 # Input image is at index 0 in batch
        target_accessor: 1 # Target dictionary is at index 1 in batch
      criterion: # Criterion adapter
        _target_: lighter.adapters.CriterionAdapter
        pred_argument: 0 # Predictions (model output) is the first argument
        target_argument: 1 # Targets (batch target) is the second argument
      metrics: # Metrics adapter
        _target_: lighter.adapters.MetricsAdapter
        pred_argument: 0
        target_argument: 1
    val: # Adapters for validation stage (same as training in this case)
      batch:
        _target_: lighter.adapters.BatchAdapter
        input_accessor: 0
        target_accessor: 1
      criterion:
        _target_: lighter.adapters.CriterionAdapter
        pred_argument: 0
        target_argument: 1
      metrics:
        _target_: lighter.adapters.MetricsAdapter
        pred_argument: 0
        target_argument: 1
    test: # Adapters for test stage (same as validation)
      batch:
        _target_: lighter.adapters.BatchAdapter
        input_accessor: 0
        target_accessor: 1
      criterion:
        _target_: lighter.adapters.CriterionAdapter
        pred_argument: 0
        target_argument: 1
```

*   **`batch` adapter**: Extracts input and target from the batch. We assume the dataloader returns `(input, target_dict)`.
*   **`criterion` and `metrics` adapters**:  Specify that predictions and targets are passed as the first and second arguments to the loss function and metrics.

## Complete Configuration (`config.yaml`)

Here is the complete `config.yaml` for multi-task learning:

```yaml title="config.yaml"
project: projects/cifar10/ # Project path

trainer:
  accelerator: "auto"
  max_epochs: 20
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: "val/classification/Accuracy/epoch" # Monitor validation classification accuracy
      mode: "max"
      filename: "best_model"

system:
  _target_: lighter.System

  model:
    _target_: models.multitask_net.MultiTaskNet # Custom multi-task model
    num_classes_classification: 10
    num_classes_segmentation: 2

  criterion:
    _target_: losses.multitask_loss.MultiTaskLoss # Custom multi-task loss
    losses:
      classification:
        _target_: torch.nn.CrossEntropyLoss
      segmentation:
        _target_: torch.nn.CrossEntropyLoss # Example: CrossEntropyLoss for segmentation
    weights:
      classification: 1.0
      segmentation: 0.5 # Lower weight for segmentation (synthetic labels)

  metrics:
    train: # Metrics for training
      classification: # Metrics for classification task
        - _target_: torchmetrics.Accuracy
          task: "multiclass"
          num_classes: 10
      segmentation: # Metrics for segmentation task
        - _target_: torchmetrics.DiceMetric
          task: "binary" # Binary segmentation
    val: # Metrics for validation
      classification:
        - _target_: torchmetrics.Accuracy
          task: "multiclass"
          num_classes: 10
      segmentation:
        - _target_: torchmetrics.DiceMetric
          task: "binary"
    test: # Metrics for test
      classification:
        - _target_: torchmetrics.Accuracy
          task: "multiclass"
          num_classes: 10
      segmentation:
        - _target_: torchmetrics.DiceMetric
          task: "binary"

  adapters:
    train: # Adapters for training stage
      batch: # Batch adapter
        _target_: lighter.adapters.BatchAdapter
        input_accessor: 0 # Input image is at index 0 in batch
        target_accessor: 1 # Target dictionary is at index 1 in batch
      criterion: # Criterion adapter
        _target_: lighter.adapters.CriterionAdapter
        pred_argument: 0 # Predictions (model output) is the first argument
        target_argument: 1 # Targets (batch target) is the second argument
      metrics: # Metrics adapter
        _target_: lighter.adapters.MetricsAdapter
        pred_argument: 0
        target_argument: 1
    val: # Adapters for validation stage (same as training in this case)
      batch:
        _target_: lighter.adapters.BatchAdapter
        input_accessor: 0
        target_accessor: 1
      criterion:
        _target_: lighter.adapters.CriterionAdapter
        pred_argument: 0
        target_argument: 1
      metrics:
        _target_: lighter.adapters.MetricsAdapter
        pred_argument: 0
        target_argument: 1
    test: # Adapters for test stage (same as validation)
      batch:
        _target_: lighter.adapters.BatchAdapter
        input_accessor: 0
        target_accessor: 1
      criterion:
        _target_: lighter.adapters.CriterionAdapter
        pred_argument: 0
        target_argument: 1
      metrics:
        _target_: lighter.adapters.MetricsAdapter
        pred_argument: 0
        target_argument: 1

  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: datasets.multitask_cifar10.build_multitask_cifar10_dataset # Custom dataset
        root: ".datasets/"
        train: true
        download: true
        transform: # Transforms for both tasks (can be task-specific if needed)
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.RandomResizedCrop
              size: 224
              scale: [0.8, 1.0]
              ratio: [0.75, 1.3333]
            - _target_: torchvision.transforms.RandomHorizontalFlip
              p: 0.5
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
      batch_size: 32
      shuffle: true
      num_workers: 4
    val: # Validation dataloader
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: datasets.multitask_cifar10.build_multitask_cifar10_dataset # Custom dataset
        root: ".datasets/"
        train: false
        download: false
        transform: # Validation transforms
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.Resize
              size: 256
            - _target_: torchvision.transforms.CenterCrop
              size: 224
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
      batch_size: 32
      num_workers: 4
    test: # Optional test dataloader (same as validation in this case)
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: datasets.multitask_cifar10.build_multitask_cifar10_dataset # Custom dataset
        root: ".datasets/"
        train: false
        download: false
        transform: # Validation transforms
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.Resize
              size: 256
            - _target_: torchvision.transforms.CenterCrop
              size: 224
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
      batch_size: 32
      num_workers: 4


optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4
  weight_decay: 1.0e-5

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: "$@trainer.max_epochs"
```

This complete configuration defines all components for multi-task learning:

*   **`project`**: Specifies the project path for dynamic module loading.
*   **`trainer`**: Configures PyTorch Lightning Trainer, monitoring validation classification accuracy.
*   **`system`**: Defines the Lighter System with:
    *   **`model`**: Custom `MultiTaskNet` model.
    *   **`criterion`**: Custom `MultiTaskLoss` with separate losses and weights for each task.
    *   **`metrics`**: Accuracy for classification and Dice metric for segmentation, organized by task.
    *   **`adapters`**: `BatchAdapter`, `CriterionAdapter`, and `MetricsAdapter` configured for multi-task data and loss/metric arguments.
    *   **`dataloaders`**: DataLoaders using the custom `MultiTaskCIFAR10` dataset.
*   **`optimizer` and `scheduler`**: AdamW optimizer and Cosine Annealing LR scheduler.

## Training and Evaluation

1.  **Create Project Files**: Create the `projects/cifar10/datasets/multitask_cifar10.py`, `projects/cifar10/models/multitask_net.py`, and `projects/cifar10/losses/multitask_loss.py` files with the code provided above.
2.  **Save Configuration**: Save the above configuration as `config_multitask.yaml`.
3.  **Start Training**: Run training using the Lighter CLI:

    ```bash title="Terminal"
    lighter fit --config config_multitask.yaml
    ```

    Monitor the training process. You should see metrics for both classification and segmentation tasks in the logs.
4.  **Evaluation**: After training, evaluate the multi-task model:

    ```bash title="Terminal"
    lighter test --config config_multitask.yaml
    ```

    This will report the test metrics for both tasks.

## Recap and Next Steps

In this tutorial, you have learned how to implement multi-task learning with Lighter. You covered:

*   Creating a custom dataset (`MultiTaskCIFAR10`) that returns data for multiple tasks.
*   Defining a multi-task model (`MultiTaskNet`) with a shared backbone and task-specific heads.
*   Implementing a custom multi-task loss function (`MultiTaskLoss`) to combine losses from different tasks.
*   Configuring metrics and adapters for multi-task learning in `config.yaml`.
*   Running training and evaluation for a multi-task model using the Lighter CLI.

This tutorial provides a basic example of multi-task learning. You can extend this approach to more complex multi-task scenarios with different tasks, datasets, and model architectures. Explore the [How-To guides](../how-to/) and [Explanation section](../explanation/) for more advanced features and customization options in Lighter.

This concludes the Tutorials section. Next, we will proceed with the How-To guides.
