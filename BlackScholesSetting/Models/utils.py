# -*- coding: utf-8 -*- 
"""
--------------------------------------------------
File Name:        utils
Description:    
Author:           jiaxuanliang
Date:             7/12/22
--------------------------------------------------
Change Activity:  7/12/22
--------------------------------------------------
"""

import numpy as np


def get_index(array, up_b, down_b):
    """
    get the largest index of a one-dimension iterative within bound
    if -1, means the boundary out of bound at the start
    """
    i_before_out = len(array)-1
    for i, a in enumerate(array):
        if a >= up_b or a <= down_b:
            i_before_out = i-1
            break
    return i_before_out


def get_indexes(ndarray, up_b, down_b):
    column_indexes = []
    for array in ndarray:
        column_indexes.append(get_index(array, up_b, down_b))
    return np.array(column_indexes)


def get_index_or_indexes(data, up_b, down_b):
    if len(data.shape) == 1:
        return get_index(data, up_b, down_b)
    else:
        return get_indexes(data, up_b, down_b)
