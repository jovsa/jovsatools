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

    # TODO: Implement for Task 3.1.
    raise NotImplementedError('Need to implement for Task 3.1')


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
