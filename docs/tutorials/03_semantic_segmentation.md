# End-to-End Semantic Segmentation

## Introduction

Semantic segmentation is a crucial computer vision task that involves classifying each pixel in an image into a predefined set of categories. This tutorial will guide you through training a semantic segmentation model using Lighter, focusing on a medical imaging dataset and the popular U-Net architecture. We will cover dataset loading, model definition, configuration, training, prediction, and visualization of results.

## Dataset: Medical Segmentation Decathlon (MSD) - Liver

For this tutorial, we will use the Liver segmentation task from the [Medical Segmentation Decathlon (MSD)](http://medicaldecathlon.com/) dataset. MSD is a collection of 10 datasets for various medical image segmentation tasks. The Liver dataset consists of CT scans and corresponding manual segmentations of the liver. We will use MONAI to easily access and process this dataset.

### Loading MSD Liver Dataset with Lighter

To load the MSD Liver dataset, we will configure the `dataloaders` section in `config.yaml` using MONAI's dataset and dataloader classes.

```yaml title="config.yaml"
system:
  dataloaders:
    train:
      _target_: monai.data.DataLoader
      dataset:
        _target_: monai.data.Dataset
        data:
          _target_: monai.apps.load_and_split_data
          dataset_name: "MSD_Liver"
          data_dir: ".datasets/"
          is_Thor=False # CT images
          val_frac: 0.2
          seed: 42
          section: "training" # "training" or "validation"
        transform:
          _target_: monai.transforms.Compose
          transforms:
            - _target_: monai.transforms.LoadImaged
              keys: ["image", "label"] # "image" and "label" fields in input dict
            - _target_: monai.transforms.AddChanneld
              keys: ["image", "label"]
            - _target_: monai.transforms.Orientationd
              keys: ["image", "label"]
              axcodes: "RAS" # Standard medical image orientation
            - _target_: monai.transforms.Spacingd
              keys: ["image", "label"]
              pixdim: [1.5, 1.5, 2.0] # Adjust spacing as needed
              mode: ["bilinear", "nearest"] # Interpolation modes
            - _target_: monai.transforms.ScaleIntensityRanged
              keys: ["image"]
              a_min: -57
              a_max: 164
              b_min: 0.0
              b_max: 1.0
              clip=True
            - _target_: monai.transforms.CropForegroundd
              keys: ["image", "label"]
              source_key: "image" # Use image to determine foreground
            - _target_: monai.transforms.RandCropByPosNegLabeld
              keys: ["image", "label"]
              label_key: "label"
              spatial_size: [96, 96, 96] # Patch size
              pos=1
              neg=1
              num_samples=4 # Number of patches per image
              image_key="image"
              image_threshold=0 # Threshold to determine valid patches
            - _target_: monai.transforms.RandAffined
              keys: ["image", "label"]
              mode= ["bilinear", "nearest"]
              prob=0.5
              spatial_dims=3
              rotate_range=null
              shear_range=null
              translate_range=null
              scale_range=null
            - _target_: monai.transforms.ToTensord
              keys: ["image", "label"]
      batch_size: 2 # Adjust based on GPU memory
      shuffle: true
      num_workers: 2 # Adjust based on system resources
```

Let's break down the key components of this dataloader configuration:

*   **`_target_: monai.data.DataLoader`**: Specifies the use of MONAI's `DataLoader`.
*   **`dataset`**: Defines the dataset using MONAI's `Dataset` class.
    *   **`data`**: Configures data loading and splitting using `monai.apps.load_and_split_data`.
        *   **`dataset_name: "MSD_Liver"`**:  Specifies the MSD Liver dataset.
        *   **`data_dir: ".datasets/"`**:  Directory to download/load the dataset.
        *   **`is_Thor=False`**:  Indicates CT images (not MRI).
        *   **`val_frac: 0.2`**:  Fraction of data to use for validation split.
        *   **`seed: 42`**:  Random seed for reproducibility.
        *   **`section: "training"`**:  Loads the training section of the split data.
    *   **`transform`**: Defines a series of data transformations using `monai.transforms.Compose`.
        *   **`LoadImaged`**: Loads image and label files.
        *   **`AddChanneld`**: Adds a channel dimension to the 3D images.
        *   **`Orientationd`**:  Standardizes image orientation to RAS.
        *   **`Spacingd`**: Resamples images to a specified spacing.
        *   **`ScaleIntensityRanged`**:  Scales intensity values to a [0, 1] range.
        *   **`CropForegroundd`**: Crops foreground region to reduce background.
        *   **`RandCropByPosNegLabeld`**: Extracts random patches, ensuring a balance of positive and negative labels.
        *   **`RandAffined`**: Applies random affine transformations for data augmentation.
        *   **`ToTensord`**: Converts data to PyTorch tensors.
*   **`batch_size: 2`**: Sets batch size (adjust based on GPU memory).
*   **`shuffle: true`**: Shuffles data.
*   **`num_workers: 2`**: Number of worker processes for data loading.

You can define a validation dataloader similarly, changing `section: "validation"` in the `load_and_split_data` configuration.

## Model: U-Net

We will use the classic U-Net architecture for semantic segmentation. MONAI provides a convenient implementation of U-Net that we can easily integrate.

### Defining U-Net in `config.yaml`

Configure the model in the `system.model` section of your `config.yaml` as follows:

```yaml title="config.yaml"
system:
  model:
    _target_: monai.networks.nets.UNet
    dimensions: 3 # 3D images
    in_channels: 1 # Input channels (1 for CT)
    out_channels: 2 # Output channels (background and liver)
    channels: [16, 32, 64, 128, 256] # Feature channels in each layer
    dropout: 0.0 # Dropout probability
    num_res_units: 2 # Number of residual units per layer
```

This configuration uses `monai.networks.nets.UNet` and sets the parameters for a 3D U-Net suitable for liver segmentation:

*   **`dimensions: 3`**: Specifies a 3D U-Net.
*   **`in_channels: 1`**: Input channels are 1 (for CT scans, which are grayscale).
*   **`out_channels: 2`**: Output channels are 2 (for background and liver classes).
*   **`channels`**: Defines the number of feature channels in each layer of the U-Net.
*   **`dropout: 0.0`**: Sets dropout probability to 0.
*   **`num_res_units: 2`**: Number of residual units in each layer.

## Loss Function and Inferer

For semantic segmentation, Dice loss is a commonly used loss function. We will also use the `SlidingWindowInferer` from MONAI for efficient inference, especially for 3D medical images.

### Configuring Loss and Inferer in `config.yaml`

Add the following to your `system` configuration in `config.yaml`:

```yaml title="config.yaml"
system:
  criterion:
    _target_: monai.losses.DiceLoss
    to_onehot_y: true # Convert labels to one-hot
    softmax: true # Apply softmax to predictions

  inferer:
    _target_: monai.inferers.SlidingWindowInferer
    roi_size: [96, 96, 96] # Region of interest size for sliding window
    sw_batch_size: 4 # Batch size for sliding window inference
    overlap: 0.5 # Overlap ratio for sliding window
```

*   **`criterion`**: Configures Dice loss using `monai.losses.DiceLoss`.
    *   **`to_onehot_y: true`**: Converts integer labels to one-hot format.
    *   **`softmax: true`**: Applies softmax activation to the model outputs.
*   **`inferer`**: Sets up `SlidingWindowInferer` for inference.
    *   **`roi_size: [96, 96, 96]`**: Defines the size of the region of interest (patch size) for sliding window inference.
    *   **`sw_batch_size: 4`**: Batch size for processing patches in the sliding window.
    *   **`overlap: 0.5`**:  Overlap between adjacent sliding windows to reduce boundary artifacts.

## Metrics and Logging

We will use the Dice metric to evaluate segmentation performance. We will also configure a `FileWriter` callback to save segmentation predictions during validation.

### Configuring Metrics and FileWriter in `config.yaml`

Add the following to your `system.metrics` and `trainer.callbacks` sections in `config.yaml`:

```yaml title="config.yaml"
system:
  metrics:
    val:
      - _target_: monai.metrics.DiceMetric
        include_background: false # Exclude background class from metric
        reduction: "mean_batch" # Reduce metric over batch
    test: # You can also define metrics for the test stage
      - _target_: monai.metrics.DiceMetric
        include_background: false
        reduction: "mean_batch"

trainer:
  callbacks:
    - _target_: lighter.callbacks.FileWriter
      path: "outputs/predictions" # Directory to save predictions
      writer: "itk_seg_nrrd" # Writer function for ITK NRRD segmentation format
```

*   **`metrics`**: Defines Dice metric for validation and test stages using `monai.metrics.DiceMetric`.
    *   **`include_background: false`**: Excludes the background class (class index 0) from the Dice metric calculation.
    *   **`reduction: "mean_batch"`**:  Averages the Dice score over the batch.
*   **`callbacks`**: Configures the `FileWriter` callback.
    *   **`_target_: lighter.callbacks.FileWriter`**: Specifies the `FileWriter` callback.
    *   **`path: "outputs/predictions"`**:  Directory where prediction files will be saved.
    *   **`writer: "itk_seg_nrrd"`**:  Specifies the `itk_seg_nrrd` writer function to save segmentations in ITK NRRD format (suitable for medical images).

## Complete Configuration (`config.yaml`)

Here is the complete `config.yaml` file for training a 3D U-Net for liver segmentation on the MSD Liver dataset:

```yaml title="config.yaml"
trainer:
  accelerator: "auto" # Use GPU if available, else CPU
  max_epochs: 20 # Train for 20 epochs
  callbacks:
    - _target_: pytorch_lightning.callbacks.ModelCheckpoint # Save best checkpoint
      monitor: "val/metrics/DiceMetric/epoch" # Monitor validation Dice metric
      mode: "max" # Save when metric is maximized
      filename: "best_model" # Checkpoint file name prefix
    - _target_: lighter.callbacks.FileWriter
      path: "outputs/predictions"
      writer: "itk_seg_nrrd"

system:
  _target_: lighter.System

  model:
    _target_: monai.networks.nets.UNet
    dimensions: 3 # 3D images
    in_channels: 1 # Input channels (1 for CT)
    out_channels: 2 # Output channels (background and liver)
    channels: [16, 32, 64, 128, 256]
    dropout: 0.0
    num_res_units: 2

  criterion:
    _target_: monai.losses.DiceLoss
    to_onehot_y: true
    softmax: true

  inferer:
    _target_: monai.inferers.SlidingWindowInferer
    roi_size: [96, 96, 96]
    sw_batch_size: 4
    overlap: 0.5

  metrics:
    val:
      - _target_: monai.metrics.DiceMetric
        include_background: false
        reduction: "mean_batch"
    test:
      - _target_: monai.metrics.DiceMetric
        include_background: false
        reduction: "mean_batch"

  dataloaders:
    train:
      _target_: monai.data.DataLoader
      dataset:
        _target_: monai.data.Dataset
        data:
          _target_: monai.apps.load_and_split_data
          dataset_name: "MSD_Liver"
          data_dir: ".datasets/"
          is_Thor=False
          val_frac: 0.2
          seed: 42
          section: "training"
        transform:
          _target_: monai.transforms.Compose
          transforms:
            - _target_: monai.transforms.LoadImaged
              keys: ["image", "label"]
            - _target_: monai.transforms.AddChanneld
              keys: ["image", "label"]
            - _target_: monai.transforms.Orientationd
              keys: ["image", "label"]
              axcodes: "RAS"
            - _target_: monai.transforms.Spacingd
              keys: ["image", "label"]
              pixdim: [1.5, 1.5, 2.0]
              mode: ["bilinear", "nearest"]
            - _target_: monai.transforms.ScaleIntensityRanged
              keys: ["image"]
              a_min: -57
              a_max: 164
              b_min: 0.0
              b_max: 1.0
              clip=True
            - _target_: monai.transforms.CropForegroundd
              keys: ["image", "label"]
              source_key: "image"
            - _target_: monai.transforms.RandCropByPosNegLabeld
              keys: ["image", "label"]
              label_key: "label"
              spatial_size: [96, 96, 96]
              pos=1
              neg=1
              num_samples=4
              image_key="image"
              image_threshold=0
            - _target_: monai.transforms.RandAffined
              keys: ["image", "label"]
              mode= ["bilinear", "nearest"]
              prob=0.5
              spatial_dims=3
              rotate_range=null
              shear_range=null
              translate_range=null
              scale_range=null
            - _target_: monai.transforms.ToTensord
              keys: ["image", "label"]
    val:
      _target_: monai.data.DataLoader
      dataset:
        _target_: monai.data.Dataset
        data:
          _target_: monai.apps.load_and_split_data
          dataset_name: "MSD_Liver"
          data_dir: ".datasets/"
          is_Thor=False
          val_frac: 0.2
          seed: 42
          section: "validation" # Use validation split
        transform: # Use same transforms as training, or adjust
          _target_: monai.transforms.Compose
          transforms:
            - _target_: monai.transforms.LoadImaged
              keys: ["image", "label"]
            - _target_: monai.transforms.AddChanneld
              keys: ["image", "label"]
            - _target_: monai.transforms.Orientationd
              keys: ["image", "label"]
              axcodes: "RAS"
            - _target_: monai.transforms.Spacingd
              keys: ["image", "label"]
              pixdim: [1.5, 1.5, 2.0]
              mode: ["bilinear", "nearest"]
            - _target_: monai.transforms.ScaleIntensityRanged
              keys: ["image"]
              a_min: -57
              a_max: 164
              b_min: 0.0
              b_max: 1.0
              clip=True
            - _target_: monai.transforms.CropForegroundd
              keys: ["image", "label"]
              source_key: "image"
            - _target_: monai.transforms.RandCropByPosNegLabeld # Optional, remove or adjust for validation
              keys: ["image", "label"]
              label_key: "label"
              spatial_size: [96, 96, 96]
              pos=1
              neg=1
              num_samples=4
              image_key="image"
              image_threshold=0
            - _target_: monai.transforms.ToTensord
              keys: ["image", "label"]
    test: # Optional test dataloader
      _target_: monai.data.DataLoader
      dataset:
        _target_: monai.data.Dataset
        data:
          _target_: monai.apps.load_and_split_data
          dataset_name: "MSD_Liver"
          data_dir: ".datasets/"
          is_Thor=False
          val_frac: 0.0 # No validation split, use full test set
          seed: 42
          section: "test" # Load test set
        transform: # Use same transforms as training, or adjust
          _target_: monai.transforms.Compose
          transforms:
            - _target_: monai.transforms.LoadImaged
              keys: ["image", "label"]
            - _target_: monai.transforms.AddChanneld
              keys: ["image", "label"]
            - _target_: monai.transforms.Orientationd
              keys: ["image", "label"]
              axcodes: "RAS"
            - _target_: monai.transforms.Spacingd
              keys: ["image", "label"]
              pixdim: [1.5, 1.5, 2.0]
              mode: ["bilinear", "nearest"]
            - _target_: monai.transforms.ScaleIntensityRanged
              keys: ["image"]
              a_min: -57
              a_max: 164
              b_min: 0.0
              b_max: 1.0
              clip=True
            - _target_: monai.transforms.CropForegroundd
              keys: ["image", "label"]
              source_key: "image" # Use image to determine foreground
            - _target_: monai.transforms.ToTensord
              keys: ["image", "label"]

optimizer:
    _target_: torch.optim.AdamW
    lr: 1.0e-4
    weight_decay: 1.0e-5
scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    T_max: "$@trainer.max_epochs" # Cosine Annealing based on max epochs

```

This complete configuration includes:

*   **`trainer`**: Configures PyTorch Lightning Trainer for 20 epochs, saves the best model checkpoint based on validation Dice metric, and sets up the `FileWriter` callback.
*   **`system`**: Defines the Lighter System with:
    *   **`model`**: 3D U-Net from MONAI.
    *   **`criterion`**: Dice loss.
    *   **`inferer`**: `SlidingWindowInferer`.
    *   **`metrics`**: Dice metric for validation and test.
    *   **`dataloaders`**: DataLoaders for train, val, and test stages using MSD Liver dataset and MONAI transforms.
*   **`optimizer`**: AdamW optimizer.
*   **`scheduler`**: Cosine Annealing learning rate scheduler.

## Training and Prediction

To start training, save the above configuration as `config_segmentation.yaml`. Then, run:

```bash title="Terminal"
lighter fit --config config_segmentation.yaml
```

Monitor the training progress in your terminal. After training, you can run prediction on the validation or test set using:

```bash title="Terminal"
lighter predict --config config_segmentation.yaml
```

The `FileWriter` callback will save the segmentation predictions in the `outputs/predictions` directory in ITK NRRD format.

## Visualization

To visualize the saved segmentation outputs, you can use medical image viewers like [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php). ITK-SNAP supports the NRRD format and allows you to overlay segmentations on the original CT images to visually assess the results.

## Recap and Next Steps

In this tutorial, you have learned how to train a 3D U-Net model for semantic segmentation of the liver using Lighter and MONAI. You covered:

*   Loading and preprocessing a medical image segmentation dataset (MSD Liver) using MONAI.
*   Configuring a 3D U-Net model, Dice loss, and `SlidingWindowInferer` in `config.yaml`.
*   Setting up metrics and a `FileWriter` callback for evaluation and saving predictions.
*   Running training and prediction using the Lighter CLI.
*   Visualizing segmentation results with ITK-SNAP.

This tutorial provides a starting point for tackling more complex medical image segmentation tasks with Lighter. In the next tutorials, we will explore [transfer learning](04_transfer_learning.md) and [multi-task learning](05_multi_task_learning.md). You can also explore the [How-To guides](../how-to/) for more specific tasks and customizations, and the [Explanation section](../explanation/) for deeper insights into Lighter's design.
