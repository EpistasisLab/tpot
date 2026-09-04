try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # for Python < 3.8

__version__ = version("tpot")
