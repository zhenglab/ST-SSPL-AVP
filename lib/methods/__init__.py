# Copyright (c) CAIRI AI Lab. All rights reserved
# Code adapted from:
# https://github.com/chengtan9907/OpenSTL

from .mgma import MGMA

method_maps = {
    'mgma': MGMA
}

__all__ = [
    'MGMA'
]