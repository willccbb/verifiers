import inspect
from typing import Callable


async def maybe_await(func: Callable, *args, **kwargs):
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)
