import verifiers as vf
from exa_env import load_environment as lxe

def load_environment(**kwargs) -> vf.Environment:
    """
    Loads a custom environment.
    """
    return lxe(**kwargs)
