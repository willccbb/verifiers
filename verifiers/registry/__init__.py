from ..envs.environment import Environment


def load_environment(env_id: str, env_args: dict) -> Environment:
    from . import sentence_repeater

    REGISTRY = {
        "sentence-repeater": sentence_repeater.load_environment,
    }
    return REGISTRY[env_id](**env_args)
