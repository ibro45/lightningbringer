from typing import Any

from pathlib import Path

import fire
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner import Tuner

from lighter.system import System
from lighter.utils.dynamic_imports import import_module_from_path
from lighter.utils.patches import PatchedConfigParser as ConfigParser
from lighter.utils.types import Mode, Stage


class Config:
    """Configuration manager for ML experiments using YAML files."""

    REQUIRED_SECTIONS = {"system", "trainer"}
    OPTIONAL_SECTIONS = {"_meta_", "_requires_", "args", "vars", "project"}
    ALLOWED_SECTIONS = REQUIRED_SECTIONS | OPTIONAL_SECTIONS
    PROHIBITED_STAGE_ARGS = {"model", "train_loaders", "validation_loaders", "dataloaders", "datamodule"}
    DATALOADER_SECTIONS = {Mode.TRAIN, Mode.VAL, Mode.TEST, Mode.PREDICT}
    METRICS_SECTIONS = {Mode.TRAIN, Mode.VAL, Mode.TEST}

    def __init__(self, config_path: str | None = None, **config_overrides: Any):
        """
        Initialize the Config object.

        Args:
            config_path: Path to the YAML configuration file
            config_overrides: Keyword arguments to override values in the configuration file
        """
        self._config_parser = ConfigParser()
        self._config_parser.read_config(config_path)
        self._config_parser.parse()
        self._config_parser.update(config_overrides)
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the configuration sections and arguments."""
        config_keys = self.get().keys()

        missing_sections = self.REQUIRED_SECTIONS - set(config_keys)
        invalid_sections = set(config_keys) - self.ALLOWED_SECTIONS

        errors = []
        if missing_sections:
            errors.append(f"Missing required sections: {missing_sections}")
        if invalid_sections:
            errors.append(f"Invalid sections found: {invalid_sections}")
        if errors:
            raise ValueError("\n".join(errors))

        self._validate_args_config()
        self._validate_system_config()

    def _validate_args_config(self) -> None:
        # TODO: Verify this method
        """Validate stage-specific arguments in the configuration."""
        args_config = self.get("args", {})

        for stage, value in args_config.items():
            if stage not in Stage:
                raise ValueError(f"Invalid stage in 'args': {stage}")

            if isinstance(value, dict):
                if prohibited := self.PROHIBITED_STAGE_ARGS.intersection(value.keys()):
                    raise ValueError(
                        f"Prohibited argument(s) in 'args#{stage}': {prohibited}. "
                        "Model and datasets should be defined within the 'system'."
                    )
            elif isinstance(value, str) and not value.startswith(("%", "@")):
                raise ValueError(f"Stage value must be dict or interpolator starting with '%' or '@', got: {value}")
            elif not isinstance(value, (dict, str)):
                raise ValueError(f"Invalid type for 'args#{stage}'. Expected dict or str, got {type(value)}")

    def _validate_system_config(self) -> None:
        # TODO: Verify this method
        """Validate system-specific sections in the configuration."""
        system_config = self.get("system", {})
        for key, valid in [("dataloaders", self.DATALOADER_SECTIONS), ("metrics", self.METRICS_SECTIONS)]:
            if key in system_config:
                if invalid := set(system_config[key].keys()) - valid:
                    raise ValueError(f"Invalid section(s) in 'system.{key}': {invalid}. Allowed sections are: {valid}")

    def get(self, key: str | None = None, default: Any = None) -> Any:
        """Get raw content for the given key. If key is None, get the entire config."""
        return self._config_parser.config if key is None else self._config_parser.config.get(key, default)

    def get_parsed_content(self, key: str | None = None, default: Any = None) -> Any:
        """
        Get the parsed content for the given key. If key is None, get the entire parsed config.
        """
        return self._config_parser.get_parsed_content(key, default=default)


class Runner:
    TRAINER_STAGE_MODES = {
        Stage.FIT: [Mode.TRAIN, Mode.VAL],
        Stage.VALIDATE: [Mode.VAL],
        Stage.TEST: [Mode.TEST],
        Stage.PREDICT: [Mode.PREDICT],
    }

    TUNER_STAGE_MODES = {
        Stage.LR_FIND: [Mode.TRAIN, Mode.VAL],
        Stage.SCALE_BATCH_SIZE: [Mode.TRAIN, Mode.VAL],
    }

    STAGE_MODES = TRAINER_STAGE_MODES | TUNER_STAGE_MODES

    def __init__(self):
        self.config = None
        self.system = None
        self.trainer = None
        self.args = None

    def _get_stage_config(self, config: dict[str, Any], stage: str) -> dict[str, Any]:
        """Get stage-specific configuration by filtering unused components."""
        stage_config = config.copy()
        system_config = stage_config.get("system", {})
        dataloader_config = system_config.get("dataloaders", {})
        metrics_config = system_config.get("metrics", {})

        # Remove dataloaders not relevant to the current stage
        for mode in set(dataloader_config) - set(self.STAGE_MODES[stage]):
            dataloader_config.pop(mode, None)

        # Remove metrics not relevant to the current stage
        for mode in set(metrics_config) - set(self.STAGE_MODES[stage]):
            metrics_config.pop(mode, None)

        # Remove optimizer, scheduler, and criterion if not relevant to the current stage
        if stage in [Stage.VALIDATE, Stage.TEST, Stage.PREDICT]:
            if stage != Stage.VALIDATE:
                system_config.pop("criterion", None)
            system_config.pop("optimizer", None)
            system_config.pop("scheduler", None)

        # Remove args not relevant to the current stage
        if "args" in stage_config:
            stage_config["args"] = {stage: stage_config["args"][stage]} if stage in stage_config["args"] else {}

        return stage_config

    def _save_config(self) -> None:
        """Save config to system checkpoint and trainer logger."""
        self.system.save_hyperparameters(self.config.get())
        if self.trainer.logger:
            self.trainer.logger.log_hyperparams(self.config.get())

    def _setup_stage(self, stage: str) -> None:
        if stage not in self.STAGE_MODES:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {list(self.STAGE_MODES.keys())}")

        if self.config is None:
            raise ValueError("Config must be loaded before setting up the stage")

        # Get and parse stage-specific configuration
        stage_config = self._get_stage_config(self.config.get(), stage)
        stage_parser = ConfigParser(stage_config)

        # Import project module if specified
        if project_path := stage_config.get("project"):
            import_module_from_path("project", Path(project_path))

        # Initialize system
        self.system = stage_parser.get_parsed_content("system")
        if not isinstance(self.system, System):
            raise ValueError("'system' must be an instance of System")

        # Initialize trainer
        self.trainer = stage_parser.get_parsed_content("trainer")
        if not isinstance(self.trainer, Trainer):
            raise ValueError("'trainer' must be an instance of Trainer")

        # Set up arguments for the stage
        self.args = stage_parser.get_parsed_content(f"args#{stage}", default={})

        # Save config to system checkpoint and trainer logger
        self._save_config()

    def _run_stage(self, stage: str) -> None:
        if stage in self.TRAINER_STAGE_MODES:
            stage_method = getattr(self.trainer, stage)
        else:
            stage_method = getattr(Tuner(self.trainer), stage)
        stage_method(self.system, **self.args)

    def run(self, stage: str, config: str | None = None, **config_overrides: Any) -> None:
        seed_everything()
        self.config = Config(config, **config_overrides)
        self._setup_stage(stage)
        self._run_stage(stage)


def cli():
    runner = Runner()

    class Commands:
        def __init__(self):
            for stage in Stage:
                setattr(self, stage.value, self._make_command(stage))

        def _make_command(self, stage: str):
            def command(config: str, **config_overrides: Any):
                return runner.run(stage=stage, config=config, **config_overrides)

            command.__name__ = stage.value
            return command

    fire.Fire(Commands())
