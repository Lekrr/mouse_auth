from functools import wraps
from typing import Callable


_all_caches = []

def cached(func: Callable):
    """
    Memoises the last returned value of <func> per DataFrame identity.

    - ``f(db)``           -> compute, store, return
    - ``f(db, cache=True)`` -> return the stored value for *this* db (or
                              compute if absent)
    """
    cache: dict = {}
    _all_caches.append(cache)

    @wraps(func)
    def wrapper(db, **kwargs):
        key = id(db)
        if kwargs.get("cache") and key in cache:
            return cache[key]
        result = func(db)
        cache[key] = result
        return result

    return wrapper


def clear_cached() -> None:
    """Drop all memoised values. Call at the start of every extraction."""
    for c in _all_caches:
        c.clear()