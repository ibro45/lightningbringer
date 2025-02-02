# Transfer Learning: Finetuning a Pre-trained Model

## Introduction

Transfer learning is a powerful technique in machine learning that allows you to leverage knowledge gained from solving one problem and apply it to a different but related problem. In deep learning, this often involves using pre-trained models, which are models trained on large datasets (e.g., ImageNet). Finetuning a pre-trained model on a new, smaller dataset can significantly reduce training time and improve performance, especially when data is limited.

This tutorial will guide you through finetuning a pre-trained image classification model using Lighter. We will use a ResNet-18 model pre-trained on ImageNet and finetune it on the CIFAR-10 dataset. We will also demonstrate how to use the `Freezer` callback in Lighter to freeze and unfreeze layers during training.

## Pre-trained Model: ResNet-18 on ImageNet

We will use a ResNet-18 model pre-trained on the [ImageNet](http://www.image-net.org/) dataset, a massive dataset of millions of labeled images. PyTorch `torchvision.models` provides easy access to pre-trained models.

### Defining the Pre-trained Model in `config.yaml`

Configure the pre-trained ResNet-18 model in the `system.model` section of your `config.yaml`:

```yaml title="config.yaml"
system:
  model:
    _target_: torchvision.models.resnet18
    pretrained: true # Load pre-trained weights from ImageNet
    num_classes: 10 # Modify the final layer for CIFAR-10 (10 classes)
```

*   **`_target_: torchvision.models.resnet18`**: Specifies ResNet-18 model from `torchvision.models`.
*   **`pretrained: true`**:  Downloads and loads pre-trained weights trained on ImageNet.
*   **`num_classes: 10`**: Modifies the final fully connected layer of ResNet-18 to output 10 classes, matching the number of classes in CIFAR-10.

By setting `pretrained: true`, PyTorch automatically downloads the pre-trained weights. Setting `num_classes` modifies the last fully connected layer to match the new task.

## Dataset: CIFAR-10

We will use the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, a standard dataset for image classification. CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. We will use `torchvision.datasets.CIFAR10` to load this dataset.

### Loading CIFAR-10 Dataset

Configure the CIFAR-10 dataset in the `system.dataloaders` section of your `config.yaml`, similar to the [Image Classification tutorial](02_image_classification.md):

```yaml title="config.yaml"
system:
  dataloaders:
    train:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ".datasets/" # Directory to download dataset
        train: true # Use training split
        download: true # Download if not exists
        transform: # Data transformations
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.RandomResizedCrop
              size: 224 # Resize images to 224x224 for ResNet-18
              scale: [0.8, 1.0]
              ratio: [0.75, 1.3333]
            - _target_: torchvision.transforms.RandomHorizontalFlip
              p: 0.5
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406] # ImageNet normalization
              std: [0.229, 0.224, 0.225] # ImageNet normalization
      batch_size: 32
      shuffle: true
      num_workers: 4
    val: # Validation dataloader
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ".datasets/"
        train: false # Use validation split
        download: false # Already downloaded
        transform: # Validation transforms (no augmentation)
          _target_: torchvision.transforms.Compose
          transforms:
            - _target_: torchvision.transforms.Resize
              size: 256 # Resize for validation
            - _target_: torchvision.transforms.CenterCrop
              size: 224 # Center crop to 224x224
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.485, 0.456, 0.406] # ImageNet normalization
              std: [0.229, 0.224, 0.225] # ImageNet normalization
      batch_size: 32
      num_workers: 4
```

Key points in the dataloader configuration:

*   **`transform`**:  We use different transforms for training and validation.
    *   **Training**: Includes `RandomResizedCrop` and `RandomHorizontalFlip` for data augmentation, followed by `ToTensor` and `Normalize` (using ImageNet stats).
    *   **Validation**: Uses `Resize` and `CenterCrop` to resize and crop images to 224x224, followed by `ToTensor` and `Normalize`.
*   **`Normalize`**:  Crucially, we use the normalization parameters (mean and std) from ImageNet, as the model was pre-trained on ImageNet.

## Optimizer and Learning Rate Scheduler

We will use AdamW optimizer and Cosine Annealing LR scheduler, similar to the previous tutorials.

### Configuring Optimizer and Scheduler

Add the optimizer and scheduler configurations to your `config.yaml`:

```yaml title="config.yaml"
optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-4 # Lower learning rate for finetuning
  weight_decay: 1.0e-5

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: "$@trainer.max_epochs"
```

Note the lower learning rate (`lr: 1.0e-4`) compared to training from scratch. Finetuning typically requires a smaller learning rate to avoid disrupting the pre-trained weights too much.

## Freezer Callback: Freezing Layers

To demonstrate freezing layers, we will use the `Freezer` callback to freeze the convolutional layers of ResNet-18 for the first few epochs and then unfreeze them.

### Configuring Freezer Callback

Add the `Freezer` callback to the `trainer.callbacks` section in `config.yaml`:

```yaml title="config.yaml"
trainer:
  max_epochs: 20
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: "val/acc/epoch" # Monitor validation accuracy
      mode: "max"
      filename: "best_model"
    - _target_: lighter.callbacks.Freezer # Freezer callback
      name_starts_with: ["conv1", "layer1", "layer2", "layer3"] # Freeze conv layers
      until_epoch: 5 # Unfreeze after 5 epochs
```

*   **`_target_: lighter.callbacks.Freezer`**: Specifies the `Freezer` callback.
*   **`name_starts_with: ["conv1", "layer1", "layer2", "layer3"]`**: Freezes layers whose names start with these prefixes, which correspond to the initial convolutional layer and the first three ResNet layers.
*   **`until_epoch: 5`**: Unfreezes these layers after epoch 5.

This configuration will freeze the specified layers for the first 5 epochs and then train all layers for the remaining epochs.

## Metrics

We will use accuracy as the evaluation metric.

### Configuring Metrics

Add the accuracy metric to the `system.metrics` section:

```yaml title="config.yaml"
system:
  metrics:
    train: # Metrics for training
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10
    val: # Metrics for validation
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10
    test: # Metrics for test
      - _target_: torchmetrics.Accuracy
        task: "multiclass"
        num_classes: 10
```

This configures `torchmetrics.Accuracy` for train, val, and test stages, suitable for multi-class classification with 10 classes.

## Complete Configuration (`config.yaml`)

Here is the complete `config.yaml` for finetuning a pre-trained ResNet-18 on CIFAR-10:

```yaml title="config.yaml"
trainer:
  accelerator: "auto"
  max_epochs: 20
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint
      monitor: "val/acc/epoch"
      mode: "max"
      filename: "best_model"
    - _target_: lighter.callbacks.Freezer
      name_starts_with: ["conv1", "layer1", "layer2", "layer3"]
      until_epoch: 5

system:
  _target_: lighter.System

  model:
    _target_: torchvision.models.resnet18
    pretrained: true
    num_classes: 10

  criterion:
    _target_: torch.nn.CrossEntropyLoss

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
        root: ".datasets/"
        train: true
        download: true
        transform:
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
    val:
      _target_: torch.utils.data.DataLoader
      dataset:
        _target_: torchvision.datasets.CIFAR10
        root: ".datasets/"
        train: false
        download: false
        transform:
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
        _target_: torchvision.datasets.CIFAR10
        root: ".datasets/"
        train: false
        download: false
        transform:
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

This configuration file defines all components for finetuning: trainer, system (model, criterion, metrics, dataloaders), optimizer, scheduler, and the Freezer callback.

## Training and Evaluation

Save the above configuration as `config_transfer_learning.yaml`. To start training, run:

```bash title="Terminal"
lighter fit --config config_transfer_learning.yaml
```

Monitor the training process. You should observe that the validation accuracy increases relatively quickly due to transfer learning. The Freezer callback will freeze the initial layers for the first 5 epochs, and then train all layers.

After training, you can evaluate the finetuned model on the test set (or validation set, as we are using validation set as test set here for simplicity):

```bash title="Terminal"
lighter test --config config_transfer_learning.yaml
```

This will evaluate the best checkpoint saved during training on the validation dataloader and report the test accuracy.

## Recap and Next Steps

In this tutorial, you have learned how to perform transfer learning and finetune a pre-trained ResNet-18 model on the CIFAR-10 dataset using Lighter. You covered:

*   Loading a pre-trained ResNet-18 model from `torchvision.models`.
*   Configuring CIFAR-10 dataset and dataloaders.
*   Setting up optimizer, learning rate scheduler, and accuracy metric.
*   Using the `Freezer` callback to freeze and unfreeze layers during training.
*   Running training and evaluation using the Lighter CLI.

This tutorial demonstrates the effectiveness of transfer learning and how Lighter simplifies the process of finetuning pre-trained models. In the next tutorial, we will explore [multi-task learning](05_multi_task_learning.md). You can also explore the [How-To guides](../how-to/) for more specific tasks and customizations, and the [Explanation section](../explanation/) for deeper insights into Lighter's design.
