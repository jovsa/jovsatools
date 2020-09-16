import numpy as np
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    MAX_DIMS,
)
from numba import njit, prange
from .tensor import Function

count = njit()(count)
index_to_position = njit()(index_to_position)
broadcast_index = njit()(broadcast_index)


@njit(parallel=True)
def _matrix_multiply(
    out, out_shape, out_strides, a, a_shape, a_strides, b, b_shape, b_strides
):

    # ASSIGN3.1
    dim1, dim2 = len(out_shape) - 2, len(out_shape) - 1
    for i in prange(len(out)):

        out_index = np.zeros(MAX_DIMS, np.int32)
        count(i, out_shape, out_index)
        o = index_to_position(out_index, out_strides)

        _, oldb = out_index[dim1], out_index[dim2]
        out_index[dim2] = 0
        a_index = np.zeros(MAX_DIMS, np.int32)
        broadcast_index(out_index, out_shape, a_shape, a_index)
        j = index_to_position(a_index, a_strides)

        out_index[dim2] = oldb
        out_index[dim1] = 0
        b_index = np.zeros(MAX_DIMS, np.int32)
        broadcast_index(out_index, out_shape, b_shape, b_index)
        k = index_to_position(b_index, b_strides)

        for off in range(a_shape[-1]):
            out[o] += a[j + off * a_strides[-1]] * b[k + off * b_strides[-2]]
    return out
    # END ASSIGN3.1


def matrix_multiply(a, b):
    ls = list(a.shape)
    assert a.shape[-1] == b.shape[-2]
    ls[-1] = b.shape[-1]
    out = a.zeros(tuple(ls))
    _matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
    return out


class MatMul(Function):
    @staticmethod
    def forward(ctx, t1, t2):
        ctx.save_for_backward(t1, t2)
        return matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx, grad_output):
        t1, t2 = ctx.saved_values
        return (
            matrix_multiply(grad_output, t2.permute(0, 2, 1)),
            matrix_multiply(t1.permute(0, 2, 1), grad_output),
        )


matmul = MatMul.apply
