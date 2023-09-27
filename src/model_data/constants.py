"""Settings for model_data module"""
from importlib.util import find_spec

# Use faster pyogrio instad of fiona if available
if find_spec('pyogrio') is not None:
    GEO_ENGINE='pyogrio'
else:
    GEO_ENGINE='fiona'
