---
title: Title
template: index.html
---

<!-- Fake title -->
#

<!-- Logo -->
<div style="display: flex; justify-content: center;"><img src="assets/images/lighter_banner.png" style="width:70%;"/></div>

<!-- pip install -->
<div style="width:70%; margin:auto; text-align:center">
</br>

```bash
pip install project-lighter
```
</div>
</br>

<!-- Body -->

**Lighter** is a PyTorch/Pytorch Lightning framework for **streamlining deep learning experiments though configuration files**.

Say goodbye to messy scripts and notebooks. Lighter is here to help you organize your experiments and make them reproducible.

## Features

1. **Structured** - the pre-defined structure ensures that your experiments are standardized, organized, and reproducible.
2. **Boilerplate-free** - forget about writing training logic over and over. `System` handles it for you and is compatible with any task, from classification to self-supervised learning.
3. **Readable** - wondering what learning rate or architecture was used? Just glance at the config.
4. **Easily modifiable** - override parameters from CLI or using another config. Leave `argparse` and hardcoding in the past.

## Comparison

<div style="display: flex; justify-content: space-between">
    <div style="width: 49%;">
        <h3 style="text-align: center">PyTorch Lightning</h3>
        ```bash title="Terminal"
        python cifar10.py
        ```
        ```py title="cifar10.py"
        from pytorch_lightning import Trainer, LightningModule
        from torch.nn import CrossEntropyLoss
        from torch.optim import Adam
        from torch.utils.data import DataLoader
        from torchvision.models import resnet18
        from torchvision.datasets import CIFAR10
        from torchvision.transforms import (ToTensor
                                            Normalize,
                                            Compose)

        class Model(LightningModule):
            def __init__(self):
                super().__init__()
                self.model = resnet18(num_classes=10)
                self.criterion = CrossEntropyLoss()
            
            def forward(self, x):
                return self.model(x)
            
            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self(x)
                loss = self.criterion(y_hat, y)
                return loss
            
            def configure_optimizers(self):
                return Adam(self.model.parameters(), lr=0.001)


        transform = Compose([
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        train_dataset = CIFAR10(
            root=".datasets",
            train=True,
            download=True,
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=512,
            shuffle=True
        )

        model = Model()
        trainer = Trainer(max_epochs=100)
        trainer.fit(model, train_loader)


        ```
    </div>

    <div style="width: 49%;">
        <h3 style="text-align: center">Lighter</h3>
        ```bash title="Terminal"
        lighter fit --config cifar10.yaml
        ```
        ```yaml title="cifar10.yaml"
        trainer:
            _target_: pytorch_lightning.Trainer
            max_epochs: 100
        
        system:
            _target_: lighter.System
            batch_size: 512

            model: torchvision.models.resnet18
            num_classes: 10

            criterion:
            _target_: torch.nn.CrossEntropyLoss

            optimizer:
            _target_: torch.optim.Adam
            params: "$@system#model.parameters()"
            lr: 0.001
            
            datasets:
                train:
                _target_: torchvision.datasets.CIFAR10
                download: True
                root: .datasets
                train: True
                transform:
                    _target_: torchvision.transforms.Compose
                    transforms:
                    - _target_: torchvision.transforms.ToTensor
                    - _target_: torchvision.transforms.Normalize
                    mean: [0.5, 0.5, 0.5]
                    std: [0.5, 0.5, 0.5]
        ```
    </div>
</div>

## Cite

If you find Lighter useful in your research or project, please consider citing it:

```bibtex
@software{lighter,
  author       = {Ibrahim Hadzic and
                  Suraj Pai and
                  Keno Bressem and
                  Hugo Aerts},
  title        = {Lighter},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.8007711},
  url          = {https://doi.org/10.5281/zenodo.8007711}
}
```
