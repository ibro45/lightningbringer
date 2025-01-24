from typing import Any

from dataclasses import dataclass, field, fields, is_dataclass

from torchmetrics import Metric, MetricCollection

from lighter.adapters import BatchAdapter, CriterionAdapter, LoggingAdapter, MetricsAdapter
from lighter.utils.data import DataLoader


def nested(cls):
    """
    Decorator to handle nested dataclass creation.
    Example:
        ```
        @nested
        @dataclass
        class Example:
            ...
        ```
    """
    original_init = cls.__init__

    def __init__(self, *args, **kwargs):
        for f in fields(cls):
            if is_dataclass(f.type) and f.name in kwargs:
                kwargs[f.name] = f.type(**kwargs[f.name])
        original_init(self, *args, **kwargs)

    cls.__init__ = __init__
    return cls


@dataclass
class Metrics:
    train: Metric | MetricCollection | None = None
    val: Metric | MetricCollection | None = None
    test: Metric | MetricCollection | None = None

    def __post_init__(self):
        self.train = self._convert_to_collection(self.train)
        self.val = self._convert_to_collection(self.val)
        self.test = self._convert_to_collection(self.test)

    def _convert_to_collection(self, x):
        if x and not isinstance(x, MetricCollection):
            return MetricCollection(x)
        return {}


@dataclass
class DataLoaders:
    train: Any | None = None
    val: Any | None = None
    test: Any | None = None
    predict: Any | None = None

    # ensure that are all lighter.utils.data.DataLoader instances
    def __post_init__(self):
        for attr in fields(self):
            if getattr(self, attr.name) is not None and not isinstance(getattr(self, attr.name), DataLoader):
                raise ValueError(f"{attr.name} must be an instance of lighter.utils.data.DataLoader.")


@dataclass
class Train:
    """Train mode sub-dataclass for Adapters."""

    batch: BatchAdapter = field(default_factory=lambda: BatchAdapter(input_accessor="input", target_accessor="target"))
    criterion: CriterionAdapter = field(default_factory=lambda: CriterionAdapter(pred_argument=0, target_argument=1))
    metrics: MetricsAdapter = field(default_factory=lambda: MetricsAdapter(pred_argument=0, target_argument=1))
    logging: LoggingAdapter = field(default_factory=LoggingAdapter)


@dataclass
class Val:
    """Val mode sub-dataclass for Adapters."""

    batch: BatchAdapter = field(default_factory=lambda: BatchAdapter(input_accessor="input", target_accessor="target"))
    criterion: CriterionAdapter = field(default_factory=lambda: CriterionAdapter(pred_argument=0, target_argument=1))
    metrics: MetricsAdapter = field(default_factory=lambda: MetricsAdapter(pred_argument=0, target_argument=1))
    logging: LoggingAdapter = field(default_factory=LoggingAdapter)


@dataclass
class Test:
    """Test mode sub-dataclass for Adapters."""

    batch: BatchAdapter = field(default_factory=lambda: BatchAdapter(input_accessor="input", target_accessor="target"))
    metrics: MetricsAdapter = field(default_factory=lambda: MetricsAdapter(pred_argument=0, target_argument=1))
    logging: LoggingAdapter = field(default_factory=LoggingAdapter)


@dataclass
class Predict:
    """Predict mode sub-dataclass for Adapters."""

    batch: BatchAdapter = field(default_factory=lambda: BatchAdapter(input_accessor="input"))
    logging: LoggingAdapter = field(default_factory=LoggingAdapter)


@nested
@dataclass
class Adapters:
    """Root configuration class for all adapters across different modes."""

    train: Train = field(default_factory=Train)
    val: Val = field(default_factory=Val)
    test: Test = field(default_factory=Test)
    predict: Predict = field(default_factory=Predict)
