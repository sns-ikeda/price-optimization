from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import TypeVar, Union

Class_or_func = TypeVar("Class_or_func")


def get_object_from_module(module_path: Union[str, Path], class_or_func_name: str) -> Class_or_func:
    """Get class or function from module"""
    module_name = str(module_path).split("/")[-1].split(".")[0]
    module = SourceFileLoader(module_name, str(module_path)).load_module()
    class_or_func = getattr(module, class_or_func_name)
    return class_or_func
