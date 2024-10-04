from . import eval_utils
from .utils import *

# If amltk is installed, import the parser
try:
    from .amltk_parser import tpot2_parser
except ImportError:
    # Handle the case when amltk is not installed
    pass
    # print("amltk is not installed. Please install it to use tpot2_parser.")
    # Optional: raise an exception or provide alternative functionality