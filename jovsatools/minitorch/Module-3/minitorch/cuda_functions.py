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

    out_index = cuda.local.array(MAX_DIMS, np.int32)
    dim1, dim2 = len(out_shape) - 2, len(out_shape) - 1
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    out_index = cuda.local.array(MAX_DIMS, np.int32)
    a_index = cuda.local.array(MAX_DIMS, np.int32)
    b_index = cuda.local.array(MAX_DIMS, np.int32)
    if i < out_size:
        count(i, out_shape, out_index)
        o = index_to_position(out_index, out_strides)

        _, oldb = out_index[dim1], out_index[dim2]
        out_index[dim2] = 0

        broadcast_index(out_index, out_shape, a_shape, a_index)
        j = index_to_position(a_index, a_strides)

        out_index[dim2] = oldb
        out_index[dim1] = 0

        broadcast_index(out_index, out_shape, b_shape, b_index)
        k = index_to_position(b_index, b_strides)

        for off in range(a_shape[-1]):
            out[o] += a[j + off * a_strides[-1]] * b[k + off * b_strides[-2]]
