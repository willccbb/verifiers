import importlib

from verifiers.envs.environment import Environment


def load_environment(env_id: str, **env_args) -> Environment:
    module_name = env_id.replace("-", "_")

    # check if installed already
    try:
        module = importlib.import_module(module_name)

        if not hasattr(module, "load_environment"):
            raise AttributeError(
                f"Module '{module_name}' does not have a 'load_environment' function"
            )

        return module.load_environment(**env_args)

    except ImportError as e:
        raise ValueError(
            f"Could not import '{env_id}' environment. Ensure the package for the '{env_id}' environment is installed."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to load environment '{env_id}': {str(e)}") from e
