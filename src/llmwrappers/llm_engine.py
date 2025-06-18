import asyncio
import inspect
import sys
from random import random
from typing import Any, Generic, Type, TypeVar

from .base_wrapper import LLMWrapper

T = TypeVar("T", bound=LLMWrapper)


def get_mro_hierarchy(cls: Type) -> tuple[Type, ...]:
    """Returns the Method Resolution Order (MRO) for a class"""
    return cls.__mro__


def get_type(obj: Any) -> Type:
    if hasattr(obj, "get_type"):
        return obj.get_type()
    else:
        return type(obj)


def find_greatest_common_ancestor(objects: list[Any]) -> Type:
    """
    Find the most specific common ancestor class for a list of objects.

    Args:
        objects (list): List of Python objects

    Returns:
        type: The most specific common ancestor class

    Example:
        >>> class Animal: pass
        >>> class Mammal(Animal): pass
        >>> class Dog(Mammal): pass
        >>> class Cat(Mammal): pass
        >>> find_greatest_common_ancestor([Dog(), Cat()])
        <class 'Mammal'>
    """
    if not objects:
        raise ValueError("List of objects cannot be empty")

    # Get the MRO (Method Resolution Order) for the first object's class
    mros = [get_mro_hierarchy(get_type(obj)) for obj in objects]

    # Find common classes among all objects
    common_classes = set(mros[0])
    for mro in mros[1:]:
        common_classes.intersection_update(mro)

    if not common_classes:
        return object  # If no common ancestor found, return base object class

    # Find the most specific common ancestor
    # This will be the class that appears earliest in the first object's MRO
    for cls in mros[0]:
        if cls in common_classes:
            return cls

    return object  # Fallback to object class if no common ancestor found


class LLMEngine(Generic[T]):
    def __init__(
        self,
        wrappers: list[T],
        max_retries: int = 3,
        backoff_exp: float = 2,
        backoff_multiplier: float = 1,
        name: str = None,
    ) -> T:
        super().__init__()
        self.wrappers = wrappers
        self.max_retries = max_retries
        self.backoff_exp = backoff_exp
        self.backoff_multiplier = backoff_multiplier
        self.name = name

    def __new__(
        cls,
        wrappers: list[T],
        max_retries: int = 3,
        backoff_exp: float = 2,
        backoff_multiplier: float = 1,
        name: str = None,
    ):
        self = super().__new__(cls)

        _wrappers = list(wrappers)
        _max_retries = max_retries
        _backoffs = {wrapper: -1 for wrapper in _wrappers}

        def select_wrapper():
            if len(_wrappers) == 0:
                raise Exception("No wrappers available")
            return sorted(_wrappers, key=lambda x: x.rate_limit.next_allowed())[0]

        def async_wrapper(name):
            async def wrapper(*args, **kwargs):
                last_exception = None

                for _ in range(_max_retries + 1):
                    selection = select_wrapper()
                    if _backoffs[selection] >= 0:
                        await asyncio.sleep(
                            random()
                            * backoff_multiplier
                            * (backoff_exp ** _backoffs[selection])
                        )

                    fn = getattr(selection, name)

                    try:
                        result = await fn(*args, **kwargs)
                        _backoffs[selection] = -1
                        return result
                    except Exception as e:
                        # print stack trace
                        import traceback

                        print(traceback.format_exc())
                        last_exception = e

                        if selection in _backoffs:
                            _backoffs[selection] += 1
                            if _backoffs[selection] == max_retries:
                                sys.stderr.write(
                                    f"Wrapper {selection.__class__.__name__} failed too many times, removing it\n"
                                )
                                _wrappers.remove(selection)
                                del _backoffs[selection]

                raise last_exception

            return wrapper

        self.select_wrapper = select_wrapper
        self.async_wrapper = async_wrapper
        return self

    def __getattr__(self, name: str):
        if name == "get_type":
            return getattr(self, name)

        selection = getattr(self, "select_wrapper")()
        result = getattr(selection, name)
        if inspect.iscoroutinefunction(result):
            return getattr(self, "async_wrapper")(name)
        else:
            return result

    def get_type(self) -> Type[T]:
        return find_greatest_common_ancestor(self.wrappers)
