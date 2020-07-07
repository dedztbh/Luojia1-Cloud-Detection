import numpy as np


def bandpass(x, lo, hi):
    shape = x.shape
    avg = (lo + hi) / 2
    diff = hi - avg
    x = np.where(abs(x - avg) <= diff, x, 0)
    x.shape = shape
    return x


def highpass(x, lo):
    shape = x.shape
    x = np.where(lo <= x, x, 0)
    x.shape = shape
    return x


def lowpass(x, hi):
    shape = x.shape
    x = np.where(x <= hi, x, 0)
    x.shape = shape
    return x
