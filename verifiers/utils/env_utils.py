from __future__ import annotations

import importlib
import inspect
import logging
from importlib.metadata import entry_points
from typing import Callable

from verifiers.envs.environment import Environment

LOGGER = logging.getLogger("verifiers.utils.env_utils")


def _call_loader(
    func: Callable[..., Environment], env_id: str, **env_args
) -> Environment:
    sig = inspect.signature(func)

    if env_args:
        LOGGER.info(
            "Using provided args: "
            + ", ".join(f"{k}={v!r}" for k, v in env_args.items())
        )

    defaults = []
    for name, p in sig.parameters.items():
        if name not in env_args and p.default is not inspect._empty:
            defaults.append(f"{name}={p.default!r}")
    if defaults:
        LOGGER.info("Using default args: " + ", ".join(defaults))

    env = func(**env_args)
    LOGGER.info(f"Successfully loaded environment '{env_id}'")
    return env


def _load_from_target_spec(target: str, env_id: str, **env_args) -> Environment:
    mod, sep, attr = target.partition(":")
    if not sep or not attr:
        raise AttributeError(f"Invalid target spec '{target}'. Expected 'module:attr'.")
    module = importlib.import_module(mod)
    func = getattr(module, attr)
    if not callable(func):
        raise TypeError(f"Target '{target}' is not callable")
    return _call_loader(func, env_id, **env_args)


def _load_via_entry_point_exact(env_id: str, **env_args) -> Environment | None:
    """Exact match on the 'verifiers' entry point name. No aliasing or splitting."""
    eps = entry_points(group="verifiers")
    matches = [ep for ep in eps if ep.name == env_id]
    if not matches:
        return None
    if len(matches) > 1:
        details = ", ".join(ep.value for ep in matches)
        raise RuntimeError(
            f"Multiple 'verifiers' entry points named '{env_id}' found: {details}"
        )
    func = matches[0].load()
    if not callable(func):
        raise TypeError(
            f"Entry point '{env_id}' did not load a callable; got {type(func)!r}"
        )
    return _call_loader(func, env_id, **env_args)


def load_environment(env_id: str, **env_args) -> Environment:
    LOGGER.info(f"Loading environment: {env_id}")

    # 1) Explicit module target: "pkg.mod:callable"
    if ":" in env_id:
        try:
            return _load_from_target_spec(env_id, env_id, **env_args)
        except Exception as e:
            LOGGER.error(f"Failed to load environment {env_id} via target spec: {e}")
            raise RuntimeError(f"Failed to load environment '{env_id}': {e}") from e

    # 2) Prefer entry points (exact match only)
    try:
        ep_env = _load_via_entry_point_exact(env_id, **env_args)
        if ep_env is not None:
            return ep_env
    except Exception as e:
        LOGGER.error(f"Failed to load environment {env_id} via entry point: {e}")
        raise RuntimeError(f"Failed to load environment '{env_id}': {e}") from e

    # 3) Back-compat fallback: import by module name (slug or namespaced ID's tail)
    module_name = env_id.split("/")[-1].replace("-", "_")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        LOGGER.error(
            f"Failed to import environment module {module_name} for env_id {env_id}: {e}"
        )
        raise ValueError(
            f"Could not import '{env_id}'. Install a package that exposes a matching "
            f"[project.entry-points.verifiers] = \"{env_id}\" entry or provide 'module:attr'."
        ) from e

    if not hasattr(module, "load_environment"):
        raise AttributeError(
            f"Module '{module_name}' has no 'load_environment'. "
            f"Prefer registering an entry point named '{env_id}' under the 'verifiers' group."
        )

    return _call_loader(getattr(module, "load_environment"), env_id, **env_args)
