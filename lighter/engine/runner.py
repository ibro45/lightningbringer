from typing import Any

from pathlib import Path

import fire
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.tuner import Tuner

from lighter.engine.config import Config
from lighter.engine.resolver import Resolver
from lighter.system import System
from lighter.utils.dynamic_imports import import_module_from_path
from lighter.utils.types.enums import Stage


class Runner:
    """
    Executes the specified stage using the validated and resolved configurations.
    """

    def __init__(self):
        self.config = None
        self.resolver = None
        self.system = None
        self.trainer = None
        self.args = None

    def run(self, stage: str, config: str | dict | None = None, **config_overrides: Any) -> None:
        """Run the specified stage with the given configuration."""
        seed_everything()
        self.config = Config(config, **config_overrides)

        # Resolves stage-specific configuration
        self.resolver = Resolver(self.config)

        # Setup stage
        self._setup_stage(stage)

        # Run stage
        self._run_stage(stage)

    def _run_stage(self, stage: str) -> None:
        """Execute the specified stage (method) of the trainer."""
        if stage in [Stage.LR_FIND, Stage.SCALE_BATCH_SIZE]:
            stage_method = getattr(Tuner(self.trainer), stage)
        else:
            stage_method = getattr(self.trainer, stage)
        stage_method(self.system, **self.args)

    def _setup_stage(self, stage: str) -> None:
        """Set up the system and trainer based on the stage-specific configuration."""
        # Get and parse stage-specific configuration
        stage_config = self.resolver.get_stage_config(stage)

        # Import project module if specified
        project_path = stage_config.get("project")
        if project_path:
            import_module_from_path("project", Path(project_path))

        # Initialize system
        self.system = stage_config.get_parsed_content("system")
        if not isinstance(self.system, System):
            raise ValueError("'system' must be an instance of System")

        # Initialize trainer
        self.trainer = stage_config.get_parsed_content("trainer")
        if not isinstance(self.trainer, Trainer):
            raise ValueError("'trainer' must be an instance of Trainer")

        # Set up arguments for the stage
        self.args = stage_config.get_parsed_content(f"args#{stage}", default={})

        # Save config to system checkpoint and trainer logger
        self._save_config()

    def _save_config(self) -> None:
        """Save config to system checkpoint and trainer logger."""
        if self.system:
            self.system.save_hyperparameters(self.config.get())
        if self.trainer and self.trainer.logger:
            self.trainer.logger.log_hyperparams(self.config.get())


def cli():
    runner = Runner()

    class Commands:
        def __init__(self):
            for stage in Stage:
                setattr(self, stage, self._make_command(stage))

        def _make_command(self, stage: Stage):
            def command(config: str, **config_overrides: Any):
                return runner.run(stage=stage, config=config, **config_overrides)

            command.__name__ = stage
            command.__doc__ = f"Run the '{stage}' stage."
            return command

    fire.Fire(Commands())
