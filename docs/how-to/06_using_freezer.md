# How to Use the Freezer Callback in Lighter

## Introduction to Freezing Layers

Freezing layers in deep learning refers to the technique of preventing the weights of certain layers in a neural network from being updated during training. This is commonly used in scenarios like:

*   **Transfer Learning**: When fine-tuning a pre-trained model on a new dataset, it's often beneficial to freeze the early layers (which have learned general features) and only train the later layers that are more specific to the new task.
*   **Fine-tuning**: In fine-tuning scenarios, you might want to freeze certain layers to preserve pre-trained knowledge while adapting other parts of the model to the new data.
*   **Regularization**: Freezing layers can act as a form of regularization, preventing overfitting by reducing the number of trainable parameters.
*   **Efficient Training**: Freezing layers can speed up training and reduce memory consumption, as gradients don't need to be computed and stored for frozen layers.

Lighter's `Freezer` callback provides a convenient and flexible way to freeze and unfreeze layers in your models during training, all configurable through your `config.yaml` file.

This how-to guide will explain how to use the `Freezer` callback in Lighter to control the training of specific layers in your models.

## Using the `Freezer` Callback

The `Freezer` callback allows you to specify which layers or parameters of your model should be frozen or unfrozen during training, and for how long.

**Configuration**:

You configure the `Freezer` callback in your `config.yaml` within the `trainer.callbacks` section:

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.Freezer # Use the Freezer callback
      name_starts_with: ["model.encoder"] # Freeze layers starting with "model.encoder"
      until_epoch: 10                   # Unfreeze after epoch 10
```

*   **`_target_: lighter.callbacks.Freezer`**: Specifies that you want to use the `Freezer` callback.
*   **`name_starts_with: ["model.encoder"]`**: This argument tells `Freezer` to freeze all parameters in your model whose names start with `"model.encoder"`. This is useful for freezing entire modules or groups of layers.
*   **`until_epoch: 10`**: Specifies that the frozen layers should be unfrozen after epoch 10 of training. You can also use `until_step` to unfreeze after a certain number of training steps.

**Freezing Strategies**:

The `Freezer` callback offers several flexible strategies for specifying which layers to freeze:

1.  **Freeze by Name Prefix (`name_starts_with`)**:

    *   Use the `name_starts_with` argument to freeze parameters whose names start with a specific prefix or list of prefixes.
    *   This is useful for freezing entire modules or groups of layers within your model.
    *   **Example**:

        ```yaml title="config.yaml"
        trainer:
          callbacks:
            - _target_: lighter.callbacks.Freezer
              name_starts_with: ["model.encoder", "model.embedding"] # Freeze encoder and embedding layers
              until_epoch: 5
        ```

        This configuration freezes all parameters in your model whose names start with either `"model.encoder"` or `"model.embedding"` until epoch 5.

2.  **Freeze by Exact Name (`names`)**:

    *   Use the `names` argument to freeze specific parameters by their exact names.
    *   This is useful for fine-grained control over freezing individual layers or parameters.
    *   **Example**:

        ```yaml title="config.yaml"
        trainer:
          callbacks:
            - _target_: lighter.callbacks.Freezer
              names: ["model.classifier.weight", "model.classifier.bias"] # Freeze classifier layer weights and bias
              until_step: 1000
        ```

        This configuration freezes only the parameters named `"model.classifier.weight"` and `"model.classifier.bias"` until training step 1000.

3.  **Exclude Layers from Freezing (`except_names`, `except_name_starts_with`)**:

    *   Use `except_names` and `except_name_starts_with` arguments to exclude specific parameters or layers from being frozen, even if they would otherwise be frozen by `name_starts_with` or `names`.
    *   This is useful for selectively unfreezing certain parts of a module that is otherwise being frozen by a broader rule.
    *   **Example**:

        ```yaml title="config.yaml"
        trainer:
          callbacks:
            - _target_: lighter.callbacks.Freezer
              name_starts_with: ["model.encoder"]           # Freeze all encoder layers
              except_name_starts_with: ["model.encoder.layer5"] # Except layers starting with "model.encoder.layer5"
              until_epoch: 7
        ```

        This configuration freezes all layers starting with `"model.encoder"` except for those whose names also start with `"model.encoder.layer5"`. This allows you to freeze most of the encoder but keep layer5 trainable.

4.  **Unfreezing after a Condition (`until_step`, `until_epoch`)**:

    *   Use `until_step` to specify the training step after which layers should be unfrozen.
    *   Use `until_epoch` to specify the epoch after which layers should be unfrozen.
    *   You can use either `until_step` or `until_epoch`, or both. If you use both, the layers will be unfrozen when either condition is met.
    *   If you don't specify `until_step` or `until_epoch`, the layers will remain frozen for the entire training process (or until manually unfrozen).
    *   **Example**:

        ```yaml title="config.yaml"
        trainer:
          callbacks:
            - _target_: lighter.callbacks.Freezer
              name_starts_with: ["model.backbone"]
              until_epoch: 5    # Unfreeze after epoch 5
              until_step: 5000  # OR after step 5000 (whichever comes first)
        ```

        This configuration unfreezes the layers starting with `"model.backbone"` after either epoch 5 or training step 5000, whichever occurs first.

**Combining Freezing Strategies**:

You can combine different freezing strategies in your `config.yaml` to achieve complex freezing schedules. For example, you can freeze the entire backbone of a model initially, then gradually unfreeze different parts of it over the course of training.

**Example: Gradual Unfreezing of Layers**

```yaml title="config.yaml"
trainer:
  callbacks:
    - _target_: lighter.callbacks.Freezer # Freezer callback for initial freezing
      name_starts_with: ["model.backbone"] # Freeze backbone initially
      until_epoch: 5                     # Unfreeze backbone after epoch 5
    - _target_: lighter.callbacks.Freezer # Second Freezer callback for gradual unfreezing
      name_starts_with: ["model.encoder.layer1", "model.encoder.layer2"] # Gradually unfreeze layer1 and layer2
      start_epoch: 5                     # Start unfreezing from epoch 5 (after backbone is unfrozen)
      unfreeze_every_n_epochs: 5        # Unfreeze every 5 epochs
```

In this advanced example, we use two `Freezer` callbacks:

1.  The first `Freezer` callback freezes the entire `"model.backbone"` until epoch 5.
2.  The second `Freezer` callback starts from epoch 5 (when the backbone is unfrozen) and gradually unfreezes layers starting with `"model.encoder.layer1"` and `"model.encoder.layer2"` every 5 epochs.

This demonstrates how you can create sophisticated layer freezing and unfreezing schedules using multiple `Freezer` callbacks in Lighter.

**Inspecting Frozen Layers**:

During training, the `Freezer` callback logs information about the freezing and unfreezing of layers. You can inspect these logs (e.g., in TensorBoard or your console output) to verify which layers are frozen at different stages of training.

**Use Cases for `Freezer` Callback**:

*   **Transfer Learning**: Freeze early layers of a pre-trained model and train only the classification head.
*   **Fine-tuning**: Gradually unfreeze layers of a pre-trained model during fine-tuning.
*   **Training Stability**: Freeze certain layers in the early stages of training to stabilize the learning process.
*   **Regularization**: Freeze layers as a form of regularization to prevent overfitting.
*   **Efficient Training**: Reduce training time and memory usage by freezing layers.

## Recap: Fine-grained Layer Control with `Freezer`

The `Freezer` callback in Lighter provides a powerful and flexible mechanism for controlling the training of specific layers in your deep learning models. By using `Freezer`, you can implement various layer freezing strategies, from simple transfer learning to complex gradual unfreezing schedules, all through your `config.yaml` file. This level of control empowers you to optimize your training process, improve model performance, and efficiently utilize pre-trained knowledge.

Next, explore the [How-To guide on Implementing a Custom Inferer](07_implementing_a_custom_inferer.md) to learn how to customize the inference process in Lighter, or return to the [How-To guides section](../how-to/) for more practical problem-solving guides. You can also go back to the [Explanation section](../explanation/) for more conceptual documentation or the [Tutorials section](../tutorials/) for end-to-end examples.
