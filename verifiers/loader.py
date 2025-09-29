import importlib
import inspect
import logging
from typing import Callable

from verifiers.envs.environment import Environment


def load_environment(env_id: str, **env_args) -> Environment:
    logger = logging.getLogger("verifiers.loader")
    logger.info("Loading environment: %s", env_id)

    module_name = env_id.replace("-", "_")
    try:
        module = importlib.import_module(module_name)

        if not hasattr(module, "load_environment"):
            raise AttributeError(
                f"Module '{module_name}' does not have a 'load_environment' function. "
                "This usually means there's a package name collision. Please either:\n"
                "1. Rename your environment (e.g. suffix with '-env')\n"
                "2. Remove unneeded files with the same name\n"
                "3. Check that you've installed the correct environment package"
            )

        env_load_func: Callable[..., Environment] = getattr(
            module, "load_environment"
        )
        sig = inspect.signature(env_load_func)
        defaults_info = []
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                if isinstance(param.default, (dict, list)):
                    defaults_info.append(f"{param_name}={param.default}")
                elif isinstance(param.default, str):
                    defaults_info.append(f"{param_name}='{param.default}'")
                else:
                    defaults_info.append(f"{param_name}={param.default}")
            else:
                defaults_info.append(f"{param_name}=<required>")

        if defaults_info:
            logger.debug("Environment defaults: %s", ", ".join(defaults_info))

        provided_params = set(env_args.keys()) if env_args else set()
        all_params = set(sig.parameters.keys())
        default_params = all_params - provided_params

        if provided_params:
            provided_values = [f"{name}={env_args[name]}" for name in provided_params]
            logger.info("Using provided args: %s", ", ".join(provided_values))

        if default_params:
            default_values = []
            for param_name in default_params:
                param = sig.parameters[param_name]
                if param.default != inspect.Parameter.empty:
                    if isinstance(param.default, str):
                        default_values.append(f"{param_name}='{param.default}'")
                    else:
                        default_values.append(f"{param_name}={param.default}")
            if default_values:
                logger.info("Using default args: %s", ", ".join(default_values))

        env_instance = env_load_func(**env_args)

        if not isinstance(env_instance, Environment):
            raise TypeError(
                f"Environment '{env_id}' returned {type(env_instance)} which is not a verifiers Environment"
            )

        logger.info("Successfully loaded environment '%s'", env_id)

        return env_instance

    except ImportError as error:
        logger.error(
            "Failed to import environment module %s for env_id %s: %s",
            module_name,
            env_id,
            error,
        )
        raise ValueError(
            f"Could not import '{env_id}' environment. Ensure the package for the '{env_id}' environment is installed."
        ) from error
    except Exception as error:  # noqa: BLE001 - propagate structured message
        logger.error(
            "Failed to load environment %s with args %s: %s",
            env_id,
            env_args,
            error,
        )
        raise RuntimeError(f"Failed to load environment '{env_id}': {error}") from error
