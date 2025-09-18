import importlib
import inspect
import logging

from verifiers.envs.environment import Environment


def load_environment(env_id: str, **env_args) -> Environment:
    # Use consistent logger naming with the rest of the verifiers package
    logger = logging.getLogger("verifiers.utils.env_utils")
    
    # Log environment loading with consistent format
    logger.info(f"Loading environment {env_id}")
    logger.info(f"Environment module name {env_id.replace('-', '_')}")
    
    # Log all provided args and kwargs
    if env_args:
        logger.info(f"Environment args provided ({len(env_args)} total): {env_args}")
    else:
        logger.info("No environment args provided, using defaults")
    
    module_name = env_id.replace("-", "_")

    try:
        module = importlib.import_module(module_name)

        if not hasattr(module, "load_environment"):
            raise AttributeError(
                f"Module '{module_name}' does not have a 'load_environment' function. "
                f"This usually means there's a package name collision. Please either:\n"
                f"1. Rename your environment (e.g. suffix with '-env')\n"
                f"2. Remove unneeded files with the same name\n"
                f"3. Check that you've installed the correct environment package"
            )

        # Get the signature of the environment's load_environment function to log defaults
        env_load_func = getattr(module, "load_environment")
        try:
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
                logger.debug(f"Environment defaults: {', '.join(defaults_info)}")
                
            if env_args:
                provided_params = set(env_args.keys())
                all_params = set(sig.parameters.keys())
                default_params = all_params - provided_params
                
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
                        logger.info(f"Using defaults for: {', '.join(default_values)}")
            elif sig.parameters:
                logger.info("All parameters will use their default values")
                
        except Exception as e:
            logger.debug(f"Could not inspect environment load function signature: {e}")

        logger.debug(f"Calling {module_name}.load_environment with {len(env_args)} arguments")
        
        env_instance = module.load_environment(**env_args)
        
        logger.info(f"Successfully loaded environment {env_id} as {type(env_instance).__name__}")
        
        return env_instance

    except ImportError as e:
        logger.error(f"Failed to import environment module {module_name} for env_id {env_id}: {str(e)}")
        raise ValueError(
            f"Could not import '{env_id}' environment. Ensure the package for the '{env_id}' environment is installed."
        ) from e
    except Exception as e:
        logger.error(f"Failed to load environment {env_id} with args {env_args}: {str(e)}")
        raise RuntimeError(f"Failed to load environment '{env_id}': {str(e)}") from e
    