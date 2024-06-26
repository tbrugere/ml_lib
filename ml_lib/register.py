from typing import overload, TypeVar, Callable, Generic, Optional, ClassVar, Self, Any

# import collections
import functools as ft

T = TypeVar("T", bound=type)
U = TypeVar("U", bound=type)

class Register(dict[str, T], Generic[T]):

    def __init__(self, t: T, name_field: str = "name"):
        """
        A class that records other classes.

        an instance of this class can be used as a decorator for classes. 
        in this case, it records that classes and its name.
        after that, one can use that instance to access said class using ```__getitem__```

        :param t T: [TODO:description]
        :param name_field str: [TODO:description]
        """
        super().__init__()
        self.t = t
        self.name_field = name_field

    def _register_module(self, module: T, name: str|None=None) -> T:
        if name is None:
            if hasattr(module, self.name_field) and getattr(module, self.name_field) is not None:
                name = getattr(module, self.name_field)
            else:  name = module.__name__
        assert name is not None

        self[name] = module
        return module

    @overload
    def __call__(self, t: U, /) -> U:
        ...

    @overload
    def __call__(self, name: str, /) -> Callable[[T], T]:
        ...

    def __call__(self, arg: str|T, /): 
        if isinstance(arg, type) and issubclass(arg, self.t):
            self._register_module(arg)
            return arg
        elif isinstance(arg, str):
            r: Callable = ft.partial(self._register_module, name=arg)
            return r
        raise ValueError(f"register got argument {arg} of type {type(arg)}")

class Loader(Generic[T]):

    register: dict[str, T]

    def __init__(self, register: dict[str, T]):
        self.register = register 

    def load_config(self, config_dict: dict|str):
        """load an object from the register using a config file

        Reads a config dictionary containing the keys
         * type: required, used to get the right object from the register
         * args: optional, passed as positional argument to the object’s constructor
         * all others: passed as kwargs to the object’s constructor

        Args:
            config_dict: dictionary serialized from configuration
        """
        if isinstance(config_dict, str):
            config_dict = dict(type=config_dict)
        assert isinstance(config_dict, dict)
        assert "type" in config_dict
        config_dict = config_dict.copy()
        name = config_dict.pop("type")
        args = config_dict.pop("args", [])
        kwargs = config_dict
        return self.register[name](*args, **kwargs)

    __call__ = load_config


class LoadableMixin():
    @classmethod 
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls(**config)

    def to_config(self) -> dict[str, Any]:
        raise NotImplementedError


def try_serializing(x):
    if isinstance(x, LoadableMixin):
        return x.to_config()
    return x
