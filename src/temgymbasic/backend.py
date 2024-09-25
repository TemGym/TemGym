# backend.py
import numpy as np
import cupy as cp

# Default backend
xp = np

def set_backend(backend: str):
    global xp
    if backend == 'gpu':
        xp = cp
    else:
        xp = np