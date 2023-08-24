from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Union
from warnings import warn

from chanfig import NestedDict


class Registry(NestedDict):
    """
    `Registry` for components.

    Notes:
        `Registry` inherits from [`NestedDict`](https://chanfig.danling.org/nested_dict/).

        Therefore, `Registry` comes in a nested structure by nature.
        You could create a sub-registry by simply calling `registry.sub_registry = Registry`,
        and access through `registry.sub_registry.register()`.

    Examples:
    ```python
    >>> registry = Registry("test")
    >>> @registry.register
    ... @registry.register("Module1")
    ... class Module:
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    >>> module = registry.register(Module, "Module2")
    >>> registry
    Registry(
      ('Module1'): <class 'danling.registry.Module'>
      ('Module'): <class 'danling.registry.Module'>
      ('Module2'): <class 'danling.registry.Module'>
    )
    >>> registry.lookup("Module")
    <class 'danling.registry.Module'>
    >>> config = {"module": {"name": "Module", "a": 1, "b": 2}}
    >>> # registry.register(Module)
    >>> module = registry.build(config["module"])
    >>> type(module)
    <class 'danling.registry.Module'>
    >>> module.a
    1
    >>> module.b
    2

    ```
    """

    override: bool = False

    def __init__(self, override: bool = False):
        super().__init__()
        self.setattr("override", override)
        warn(
            "DanLing Registry has been deprecated in favor of CHANfiG Registry, and will be removed in 0.2.0.",
            category=DeprecationWarning,
            stacklevel=2,
        )

    def register(self, component: Optional[Callable] = None, name: Optional[str] = None) -> Callable:
        r"""
        Register a new component.

        Args:
            component: The component to register.
            name: The name of the component.

        Returns:
            component: The registered component.

        Raises:
            ValueError: If the component with the same name already registered and `Registry.override=False`.

        Examples:
            >>> registry = Registry("test")
            >>> @registry.register
            ... @registry.register("Module1")
            ... class Module:
            ...     def __init__(self, a, b):
            ...         self.a = a
            ...         self.b = b
            >>> module = registry.register(Module, "Module2")
            >>> registry
            Registry(
              ('Module1'): <class 'danling.registry.Module'>
              ('Module'): <class 'danling.registry.Module'>
              ('Module2'): <class 'danling.registry.Module'>
            )
        """

        if name in self and not self.override:
            raise ValueError(f"Component with name {name} already registered.")

        # Registry.register()
        if name is not None:
            self.set(name, component)

        # @Registry.register()
        @wraps(self.register)
        def register(component, name=None):
            if name is None:
                name = component.__name__
            self.set(name, component)
            return component

        # @Registry.register
        if callable(component) and name is None:
            return register(component)

        return lambda x: register(x, component)

    def lookup(self, name: str) -> Any:
        r"""
        Lookup for a component.

        Args:
            name:

        Returns:
            (Any): The component.

        Raises:
            KeyError: If the component is not registered.

        Examples:
            >>> registry = Registry("test")
            >>> @registry.register
            ... class Module:
            ...     def __init__(self, a, b):
            ...         self.a = a
            ...         self.b = b
            >>> registry.lookup("Module")
            <class 'danling.registry.Module'>
        """

        return self[name]

    def build(self, name: Union[str, Mapping], *args, **kwargs) -> Any:
        r"""
        Build a component.

        Args:
            name (str | Mapping):
                If its a `Mapping`, it must contain `"name"` as a member, the rest will be treated as `**kwargs`.
                Note that values in `kwargs` will override values in `name` if its a `Mapping`.
            *args: The arguments to pass to the component.
            **kwargs: The keyword arguments to pass to the component.

        Returns:
            (Any):

        Raises:
            KeyError: If the component is not registered.

        Examples:
            >>> registry = Registry("test")
            >>> @registry.register
            ... class Module:
            ...     def __init__(self, a, b):
            ...         self.a = a
            ...         self.b = b
            >>> config = {"module": {"name": "Module", "a": 1, "b": 2}}
            >>> # registry.register(Module)
            >>> module = registry.build(**config["module"])
            >>> type(module)
            <class 'danling.registry.Module'>
            >>> module.a
            1
            >>> module.b
            2
            >>> module = registry.build(config["module"], a=2)
            >>> module.a
            2
        """

        if isinstance(name, Mapping):
            name = deepcopy(name)
            name, kwargs = name.pop("name"), dict(name, **kwargs)  # type: ignore
        return self.lookup(name)(*args, **kwargs)  # type: ignore

    def __wrapped__(self, *args, **kwargs):
        pass


GlobalRegistry = Registry()