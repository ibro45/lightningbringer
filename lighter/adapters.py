from typing import Any, Callable

from functools import wraps

from torch.utils.data import Dataset


def compose_adapters(adapters: Callable) -> Callable:
    """
    Composes multiple adapter functions into a single adapter.

    Args:
        *adapters: A sequence of adapter functions to compose.
            Each adapter should accept a function and return a modified function.

    Returns:
        Callable: A composed adapter function that applies the input adapters in sequence.
    """

    def composed_adapter(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func
            for adapter in reversed(adapters):
                result = adapter(result)
            return result(*args, **kwargs)

        return wrapper

    return composed_adapter


def structure_batch(
    input: int | str | Callable, target: int | str | Callable | None = None, id: int | str | Callable | None = None
):
    """
    Decorator that structures batch data returned by a dataset by specifying
    how to access input, target (optional), and id (optional) fields.

    Args:
        input: Specifier for accessing input data. Can be:
            - int: Index for accessing list/tuple elements
            - str: Key for accessing dictionary elements
            - Callable: Function to transform/extract input data. Must receive a single argument, the batch.
        target: Specifier for accessing target data. Same types as input.
        id: Specifier for accessing sample ID data. Same types as input.

    Returns:
        Callable: A decorator that wraps the dataset class with the BatchStructurer functionality.
    """

    def decorator(dataset):
        return BatchStructurerAdapter(dataset, input, target, id)

    return decorator


def postprocess_data(
    input: Callable | list[Callable] | None = None,
    target: Callable | list[Callable] | None = None,
    pred: Callable | list[Callable] | None = None,
) -> Callable:
    """
    Decorator to apply post-processing function(s) to each item before passing it to the callable.

    Args:
        input: Function or list of functions to process input data
        target: Function or list of functions to process target data
        pred: Function or list of functions to process prediction data

    Returns:
        Callable: The decorated function
    """
    adapter = PostprocessAdapter(input, target, pred)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return adapter(func, *args, **kwargs)

        return wrapper

    return decorator


def map_data_to_arguments(
    input: str | int | None = None, target: str | int | None = None, pred: str | int | None = None, id: str | int | None = None
) -> Callable:
    """
    Decorator that maps the Lighter objects "input", "target", "pred", and "id"
    to their corresponding argument names or positions in the decorated function.

    Args:
        input: The argument name or position to map the `input` data to.
        target: The argument name or position to map the `target` data to.
        pred: The argument name or position to map the `pred` data to.
        id: The argument name or position to map the `id` data to.

    Returns:
        Callable: The decorated function
    """
    adapter = MapDataToArgumentsAdapter(input, target, pred, id)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return adapter(func, *args, **kwargs)

        return wrapper

    return decorator


class BatchStructurerAdapter(Dataset):
    """
    A dataset wrapper that structures batch data by specifying how to access input, target, and id fields.
    Args:
        dataset: The dataset to wrap
        input: Specifier for accessing input data. Can be:
            - int: Index for accessing list/tuple elements
            - str: Key for accessing dictionary elements
            - Callable: Function to transform/extract input data. Must receive a single argument, the batch.
        target: Specifier for accessing target data. Same types as input.
        id: Specifier for accessing sample ID data. Same types as input.
    """

    def __init__(
        self,
        dataset: Dataset,
        input: int | str | Callable,
        target: int | str | Callable | None = None,
        id: int | str | Callable | None = None,
    ):
        self.dataset = dataset
        self.input = input
        self.target = target
        self.id = id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = self.dataset[index]
        if not isinstance(data, (dict, tuple, list)):
            raise ValueError(f"Supports data of type dict, tuple, or list, got: {type(data)}")

        return {
            "input": self._get_value(data, self.input),
            "target": self._get_value(data, self.target) if self.target is not None else None,
            "id": self._get_value(data, self.id) if self.id is not None else None,
        }

    def _get_value(self, data: Any, accessor: int | str | Callable) -> Any:
        if isinstance(accessor, int) and isinstance(data, (tuple, list)):
            return data[accessor]
        elif isinstance(accessor, str) and isinstance(data, dict):
            return data.get(accessor)
        elif callable(accessor):
            return accessor(data)
        else:
            raise ValueError(f"Invalid accessor type or data structure for accessor: {accessor}")


class MapDataToArgumentsAdapter:
    """
    Adaptor class that maps the Lighter objects "input", "target", "pred", and "id"
    to their corresponding argument names in the decorated function.

    Args:
        input: The argument name or position to map the `input` data to.
        target: The argument name or position to map the `target` data to.
        pred: The argument name or position to map the `pred` data to.
        id: The argument name or position to map the `id` data to.
    """

    def __init__(
        self,
        input: str | int | None = None,
        target: str | int | None = None,
        pred: str | int | None = None,
        id: str | int | None = None,
    ):
        values = [input, target, pred, id]
        if len(set(values)) == 1 and None in values:
            raise ValueError("At least one argument must be specified.")
        non_none_values = [v for v in values if v is not None]
        if len(non_none_values) != len(set(non_none_values)):
            raise ValueError("Duplicate values are not allowed.")

        self.input = input
        self.target = target
        self.pred = pred
        self.id = id

    def __call__(self, func: Callable, *args, **kwargs) -> Any:
        """
        Applies argument mapping to the given function and arguments.

        Args:
            func: The function to call.
            *args: Positional arguments to the function.
            **kwargs: Keyword arguments to the function.

        Returns:
            The result of the function call.
        """
        # print("Original args:", args)
        # print("Original kwargs:", kwargs)
        args = list(args)
        new_kwargs = {}
        for name, map_to in [("input", self.input), ("target", self.target), ("pred", self.pred), ("id", self.id)]:
            if name in kwargs:
                if isinstance(map_to, str):
                    new_kwargs[map_to] = kwargs[name]
                elif isinstance(map_to, int):
                    if map_to >= len(args):
                        args.extend([None] * (map_to + 1 - len(args)))
                    args[map_to] = kwargs[name]
        # print("New kwargs:", new_kwargs)
        new_kwargs.update({k: v for k, v in kwargs.items() if k not in ["input", "target", "pred", "id"]})
        # print("Updated kwargs:", new_kwargs)
        return func(*args, **new_kwargs)


class PostprocessAdapter:
    """
    Adaptor class to apply post-processing function(s) to each item before passing it to the callable.

    Args:
        input: Function or list of functions to process input data
        target: Function or list of functions to process target data
        pred: Function or list of functions to process prediction data
    """

    def __init__(
        self,
        input: Callable | list[Callable] | None = None,
        target: Callable | list[Callable] | None = None,
        pred: Callable | list[Callable] | None = None,
    ):
        self.input = input
        self.target = target
        self.pred = pred

    def __call__(self, func: Callable, *args, **kwargs) -> Any:
        """
        Applies post-processing to the given function and arguments.

        Args:
            func: The function to call.
            *args: Positional arguments to the function.
            **kwargs: Keyword arguments to the function.

        Returns:
            The result of the function call.
        """
        for name, processors in [("input", self.input), ("target", self.target), ("pred", self.pred)]:
            if name in kwargs and processors is not None:
                kwargs[name] = self._apply_processors(kwargs[name], processors)

        return func(*args, **kwargs)

    def _apply_processors(data: Any, processors: Callable | list[Callable] | None) -> Any:
        # If no processors are specified, return the data unchanged
        if processors is None:
            return data
        # If processors is a list, apply each processor function sequentially
        if isinstance(processors, list):
            for proc in processors:
                data = proc(data)
            return data
        # If processors is a single function, apply it directly
        return processors(data)
