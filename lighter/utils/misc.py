"""
This module contains miscellaneous utility functions for handling lists, attributes, and function arguments.
"""

from typing import Any, Callable, List

import inspect

from torch.optim.optimizer import Optimizer


def ensure_list(input: Any) -> List:
    """
    Ensures that the input is wrapped in a list. If the input is None, returns an empty list.

    Args:
        input: The input to wrap in a list.

    Returns:
        List: The input wrapped in a list, or an empty list if input is None.
    """
    if isinstance(input, list):
        return input
    if isinstance(input, tuple):
        return list(input)
    if input is None:
        return []
    return [input]


def setattr_dot_notation(obj: Callable, attr: str, value: Any) -> None:
    """
    Sets an attribute on an object using dot notation.

    Args:
        obj: The object on which to set the attribute.
        attr: The attribute name, which can use dot notation for nested attributes.
        value: The value to set the attribute to.
    """
    if "." not in attr:
        if not hasattr(obj, attr):
            raise AttributeError(f"`{get_name(obj, True)}` has no attribute `{attr}`.")
        setattr(obj, attr, value)
    # Solve recursively if the attribute is defined in dot-notation
    else:
        obj_name, attr = attr.split(".", maxsplit=1)
        setattr_dot_notation(getattr(obj, obj_name), attr, value)


def hasarg(_callable: Callable, arg_name: str) -> bool:
    """
    Checks if a callable (function, method, or class) has a specific argument.

    Args:
        _callable: The callable to inspect.
        arg_name: The name of the argument to check for.

    Returns:
        bool: True if the argument exists, False otherwise.
    """
    args = inspect.signature(_callable).parameters.keys()
    return arg_name in args


def get_name(_callable: Callable, include_module_name: bool = False) -> str:
    """
    Retrieves the name of a callable, optionally including the module name.

    Args:
        _callable: The callable whose name to retrieve.
        include_module_name: Whether to include the module name in the result.

    Returns:
        str: The name of the callable, optionally prefixed with the module name.
    """
    # Get the name directly from the callable's __name__ attribute
    name = getattr(_callable, "__name__", type(_callable).__name__)

    if include_module_name:
        # Get the module name directly from the callable's __module__ attribute
        module = getattr(_callable, "__module__", type(_callable).__module__)
        name = f"{module}.{name}"

    return name


def get_optimizer_stats(optimizer: Optimizer) -> dict[str, float]:
    """
    Extract learning rates and momentum values from a PyTorch optimizer.

    Collects learning rate and momentum/beta values from each parameter group
    in the optimizer and returns them in a dictionary. Keys are formatted to show
    the optimizer type and group number (if multiple groups exist).

    Args:
        optimizer: The PyTorch optimizer to extract values from.

    Returns:
        dict[str, float]: dictionary containing:
            - Learning rates: "optimizer/{name}/lr[/group{N}]"
            - Momentum values: "optimizer/{name}/momentum[/group{N}]"

            Where [/group{N}] is only added for optimizers with multiple groups.
    """
    stats_dict = {}
    for group_idx, group in enumerate(optimizer.param_groups):
        lr_key = f"optimizer/{optimizer.__class__.__name__}/lr"
        momentum_key = f"optimizer/{optimizer.__class__.__name__}/momentum"

        # Add group index to the key if there are multiple parameter groups
        if len(optimizer.param_groups) > 1:
            lr_key += f"/group{group_idx+1}"
            momentum_key += f"/group{group_idx+1}"

        # Extracting learning rate
        stats_dict[lr_key] = group["lr"]

        # Extracting momentum or betas[0] if available
        if "momentum" in group:
            stats_dict[momentum_key] = group["momentum"]
        if "betas" in group:
            stats_dict[momentum_key] = group["betas"][0]

    return stats_dict
