import numpy as np


def resize_array(array, new_w, new_h):
    scale_0 = int(new_w/array.shape[0])
    scale_1 = int(new_h/array.shape[1])
    return array.repeat(scale_0, axis=0).repeat(scale_1, axis=1)
