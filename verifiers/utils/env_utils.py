import importlib
import logging

from verifiers.envs.environment import Environment


def load_environment(env_id: str, **env_args) -> Environment:
    logger = logging.getLogger(__name__)
    
    # Log environment loading details
    logger.info(f"Loading environment: '{env_id}'")
    logger.info(f"Environment module name: '{env_id.replace('-', '_')}'")
    
    # Log all args and kwargs with detailed formatting
    if env_args:
        logger.info("Environment configuration:")
        for key, value in env_args.items():
            # Format different types appropriately for logging
            if isinstance(value, (dict, list)):
                logger.info(f"  {key}: {value}")
            elif isinstance(value, str):
                logger.info(f"  {key}: '{value}'")
            else:
                logger.info(f"  {key}: {value}")
    else:
        logger.info("Environment configuration: Using all default values (no args provided)")
    
    module_name = env_id.replace("-", "_")

    # check if installed already
    try:
        module = importlib.import_module(module_name)

        if not hasattr(module, "load_environment"):
            raise AttributeError(
                f"Module '{module_name}' does not have a 'load_environment' function"
            )

        # Log before calling the environment's load_environment function
        logger.debug(f"Calling {module_name}.load_environment() with {len(env_args)} arguments")
        
        env_instance = module.load_environment(**env_args)
        
        # Log successful environment creation with type information
        logger.info(f"Successfully created environment instance: {type(env_instance).__name__}")
        logger.debug(f"Environment instance type: {type(env_instance)}")
        
        return env_instance

    except ImportError as e:
        logger.error(f"Failed to import environment module '{module_name}' for env_id '{env_id}'")
        logger.error(f"ImportError details: {str(e)}")
        raise ValueError(
            f"Could not import '{env_id}' environment. Ensure the package for the '{env_id}' environment is installed."
        ) from e
    except Exception as e:
        logger.error(f"Failed to load environment '{env_id}' with args: {env_args}")
        logger.error(f"Error details: {str(e)}")
        raise RuntimeError(f"Failed to load environment '{env_id}': {str(e)}") from e