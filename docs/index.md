---
title: Lighter
---

<!-- Fake title -->
#

<!-- Remove content from the left bar (otherwise there's "Home" just sitting there) -->
<style>
/* Navigation sidebar */
.md-nav--primary {
  display: none;
}
</style>


<!-- Logo -->
<div style="display: flex; justify-content: center;"><img src="assets/images/lighter_banner.png" style="width:50%;"/></div>

<!-- pip install -->
<div style="width:50%; margin:auto; text-align:center">
</br>

```bash
pip install lighter-framework
```
</div>
</br>

<!-- Body -->


<div class="grid cards" markdown>

-   :material-cube-outline:{ .lg .middle }  __Task-agnostic__

    ---

    Classification, segmentation, or self-supervised learning? Lighter handles it all.

-   :material-cog-outline:{ .lg .middle }  __Configuration-based__

    ---

    Define, adjust, and reproduce experiments through configuration files.

-   :material-puzzle-outline:{ .lg .middle }  __Customizable__

    ---

    Integrate custom code seamlessly, whether it's models, datasets, or any other component.

</div>


## Lighter vs. PyTorch Lightning

See how training a model on CIFAR-10 differs between Lighter and PyTorch Lightning.

=== "Lighter"
    ```bash title="Terminal"
    lighter fit config.yaml
    ```

    ```yaml title="config.yaml"
    trainer:
    _target_: pytorch_lightning.Trainer
    max_epochs: 2

    system:
        _target_: lighter.System

        model:
            _target_: torchvision.models.resnet18
            num_classes: 10

        criterion:
            _target_: torch.nn.CrossEntropyLoss

        optimizer:
            _target_: torch.optim.Adam
            params: "$@system#model.parameters()"
            lr: 0.001

        dataloaders:
            train:
                _target_: torch.utils.data.DataLoader
                batch_size: 32
                shuffle: True
                dataset:
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

=== "PyTorch Lightning"
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
    from torchvision.transforms import ToTensor, Normalize, Compose


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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = Model()
    trainer = Trainer(max_epochs=2)
    trainer.fit(model, train_loader)
    ```

## Next Steps

<div class="grid cards" markdown>

-   :material-book-open:{ .lg .middle }  __Tutorials__

    ---

    Step-by-step guides to help you get started with Lighter and run your first experiments
    [:octicons-arrow-right-24: Start Learning](tutorials/01_configuration_basics.md)

-   :material-hammer-wrench:{ .lg .middle }  __How-To Guides__
    
    ---
    
    Detailed instructions for using Lighter's advanced features and solving specific problems
    [:octicons-arrow-right-24: Learn More](how-to/01_overview.md)

-   :material-lightbulb:{ .lg .middle }  __Explanation__

    ---

    Comprehensive understanding of Lighter's design principles and architecture
    [:octicons-arrow-right-24: Explore](explanation/01_overview.md)

<!-- -   :material-api:{ .lg .middle }  __API Reference__
    
    ---
    
    Detailed documentation of Lighter's classes, functions, and interfaces
    [:octicons-arrow-right-24: View API](api/lighter.md) -->

</div>
 

## Cite

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