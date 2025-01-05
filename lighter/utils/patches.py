"""
Contains code that patches certain issues from other libraries that we expect will be resolved in the future.
"""
from typing import Any, Callable

import monai
from monai.bundle import ConfigParser
from torch.nn import ModuleDict


class PatchedModuleDict(ModuleDict):
    """
    This class provides a workaround for key conflicts in PyTorch's ModuleDict by ensuring unique internal keys.
    """

    # https://github.com/pytorch/pytorch/issues/71203
    def __init__(self, modules=None):
        """
        Initializes the PatchedModuleDict with optional modules.

        Args:
            modules (dict, optional): A dictionary of modules to initialize the ModuleDict.
        """
        self._key_map = {}
        super().__init__(modules)

    def __setitem__(self, key, module):
        """
        Sets the module for the given key, ensuring a unique internal key.

        Args:
            key (str): The key to associate with the module.
            module (torch.nn.Module): The module to store.
        """
        internal_key = f"_{key}"
        while internal_key in self._modules:
            internal_key = f"_{internal_key}"
        self._key_map[key] = internal_key
        super().__setitem__(internal_key, module)

    def __getitem__(self, key):
        """
        Retrieves the module associated with the given key.

        Args:
            key (str): The key for which to retrieve the module.

        Returns:
            torch.nn.Module: The module associated with the key.
        """
        internal_key = self._key_map.get(key, key)
        return super().__getitem__(internal_key)

    def __delitem__(self, key):
        """
        Deletes the module associated with the given key.

        Args:
            key (str): The key for which to delete the module.
        """
        internal_key = self._key_map.pop(key, key)
        super().__delitem__(internal_key)

    def __contains__(self, key):
        """
        Checks if a module is associated with the given key.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        internal_key = self._key_map.get(key, key)
        return super().__contains__(internal_key)

    def keys(self):
        """
        Returns the keys of the modules.

        Returns:
            KeysView: A view of the keys in the dictionary.
        """
        return self._key_map.keys()

    def items(self):
        """
        Returns the items (key, module) in the dictionary.

        Returns:
            Generator: A generator yielding key, module pairs.
        """
        return ((key, self._modules[internal_key]) for key, internal_key in self._key_map.items())

    def values(self):
        """
        Returns the modules in the dictionary.

        Returns:
            Generator: A generator yielding modules.
        """
        return (self._modules[internal_key] for internal_key in self._key_map.values())


class PatchedConfigParser(ConfigParser):
    """
    Patched version of MONAI's ConfigParser to incorporate the provided
    `instantiate` and `_get_wrapper` methods into the `ConfigComponent` class.

    This allows for specifying a wrapper function or class within the configuration
    itself, which will be applied to the instantiated object.
    """

    def __init__(
        self,
        config: dict | list | str | None = None,
        fnames: list[str] | str | None = None,
        globals: dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, fnames, globals, **kwargs)

        # Apply the patches to ConfigComponent within this instance
        self._patch_config_component()

    def _patch_config_component(self) -> None:
        """Patches the `ConfigComponent` class with the custom methods."""

        def _get_wrapper(self) -> None | Callable[[object], object]:
            """Check if a wrapper is specified in the config."""
            wrapper_config = self.get_config().get("_wrapper_", None)
            if wrapper_config is not None:
                if callable(wrapper_config):
                    return wrapper_config
                elif isinstance(wrapper_config, dict) and "_target_" in wrapper_config:
                    # Extract wrapper arguments
                    wrapper_args = {k: v for k, v in wrapper_config.items() if k not in ["_target_", "_mode_"]}
                    mode = wrapper_config.get("_mode_", "default")
                    wrapper = monai.utils.instantiate(wrapper_config["_target_"], mode, **wrapper_args)

                    return wrapper
                else:
                    raise ValueError(
                        f"wrapper must be a callable or a config dict, but got type {type(wrapper_config)}: {wrapper_config}. "
                        "Make sure all references are resolved before calling instantiate."
                    )
            return None

        def instantiate(self, **kwargs: Any) -> object:
            """Instantiate component with wrapper support."""
            if not self.is_instantiable(self.get_config()) or self.is_disabled():
                return None

            modname = self.resolve_module_name()
            mode = self.get_config().get("_mode_", "default")
            args = {
                k: v
                for k, v in self.get_config().items()
                if k not in ["_target_", "_disabled_", "_requires_", "_desc_", "_mode_", "_wrapper_"]
            }
            args.update(kwargs)

            wrapper = self._get_wrapper()
            try:
                instance = monai.utils.instantiate(modname, mode, **args)
                return wrapper(instance) if wrapper is not None else instance
            except Exception as e:
                if wrapper is not None:
                    raise RuntimeError(
                        f"Failed to instantiate {self}. If using a class as wrapper, make sure to set `_mode_: partial`"
                    ) from e
                raise

        # Access ConfigComponent directly and patch it
        from monai.bundle import ConfigComponent

        ConfigComponent._get_wrapper = _get_wrapper
        ConfigComponent.instantiate = instantiate
