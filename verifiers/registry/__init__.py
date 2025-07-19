import importlib
import importlib.util

from ..envs.environment import Environment


def load_environment(env_id: str, **env_args) -> Environment:
    module_name = env_id.replace("-", "_")

    try:
        module = importlib.import_module(
            f".{module_name}", package="verifiers.registry"
        )

        if not hasattr(module, "load_environment"):
            raise AttributeError(
                f"Module '{module_name}' does not have a 'load_environment' function"
            )

        return module.load_environment(**env_args)

    except ImportError as e:
        raise ValueError(
            f"Environment '{env_id}' not found. Make sure '{module_name}.py' exists in the registry folder."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to load environment '{env_id}': {str(e)}") from e


def load_eval(env_id: str, **env_args) -> Environment:
    module_name = env_id.replace("-", "_")
    module = importlib.import_module(
        f".{module_name}", package="verifiers.registry.evals"
    )
    if not hasattr(module, "load_eval"):
        raise AttributeError(
            f"Module '{module_name}' does not have a 'load_eval' function"
        )
    return module.load_eval(**env_args)


def load_local_environment(file_path: str, **env_args) -> Environment:
    try:
        spec = importlib.util.spec_from_file_location("local_env", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from '{file_path}'")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "load_environment"):
            raise AttributeError(
                f"Module at '{file_path}' does not have a 'load_environment' function"
            )

        return module.load_environment(**env_args)

    except ImportError as e:
        raise ValueError(
            f"Could not load environment from '{file_path}': {str(e)}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to load local environment from '{file_path}': {str(e)}"
        ) from e
