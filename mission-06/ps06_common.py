"""
Mission 06 - shared helper.
Station files start with a digit (01_..., 02_...), so they cannot be imported with `import`.
`load_sibling` loads one of them by file name, relative to this folder.
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_sibling(filename: str, name: str):
    """Import a station file that sits next to this module."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # needed so @dataclass can resolve the module
    spec.loader.exec_module(mod)
    return mod
