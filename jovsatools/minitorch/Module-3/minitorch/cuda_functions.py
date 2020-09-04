from numba import cuda

from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    MAX_DIMS,
)
import numpy
import numpy as np
from .tensor import Function

count = cuda.jit(device=True)(count)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)


@cuda.jit()
def _matrix_multiply(
    out, out_shape, out_strides, out_size, a, a_shape, a_strides, b, b_shape, b_strides
):

    # TODO: Implement for Task 3.4.
    raise NotImplementedError('Need to implement for Task 3.4')
    # TODO: Implement for Task 3.4.
    raise NotImplementedError('Need to implement for Task 3.4')
