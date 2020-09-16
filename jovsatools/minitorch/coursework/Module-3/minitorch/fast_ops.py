import numpy as np
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)
from numba import njit, prange

count = njit()(count)
index_to_position = njit()(index_to_position)
broadcast_index = njit()(broadcast_index)


def tensor_map(fn):
    """
    Higher-order tensor map function.

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        if (
            len(out_strides) != len(in_strides)
            or (out_strides != in_strides).any()
            or (out_shape != in_shape).any()
        ):
            for i in prange(len(out)):
                out_index = np.zeros(MAX_DIMS, np.int32)
                in_index = np.zeros(MAX_DIMS, np.int32)
                count(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                o = index_to_position(out_index, out_strides)
                j = index_to_position(in_index, in_strides)
                out[o] = fn(in_storage[j])
        else:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])


    return njit(parallel=True)(_map)


def map(fn):
    f = tensor_map(njit()(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    Higher-order tensor zipWith (or map2) function.

    Args:
        fn: function mappings two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.
    """

    def _zip(out, out_shape, out_strides, a, a_shape, a_strides, b, b_shape, b_strides):
        if (
            len(out_strides) != len(a_strides)
            or (out_strides != a_strides).any()
            or (out_shape != a_shape).any()
            or len(out_strides) != len(b_strides)
            or (out_strides != b_strides).any()
            or (out_shape != b_shape).any()
        ):
            for i in prange(len(out)):
                out_index = np.zeros(MAX_DIMS, np.int32)
                a_index = np.zeros(MAX_DIMS, np.int32)
                b_index = np.zeros(MAX_DIMS, np.int32)
                count(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                j = index_to_position(a_index, a_strides)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                k = index_to_position(b_index, b_strides)
                out[o] = fn(a[j], b[k])
        else:
            for i in prange(len(out)):
                out[i] = fn(a[i], b[i])

    return njit(parallel=True)(_zip)


def zip(fn):

    f = tensor_zip(njit()(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    Higher-order tensor reduce function.

    Args:
        fn: reduction function mapping two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_shape (array): shape of reduction (1 for dimension kept, shape value for dimensions summed out)
        reduce_size (int): size of reduce shape
    """

    def _reduce(
        out, out_shape, out_strides, a, a_shape, a_strides, reduce_shape, reduce_size
    ):
        for i in prange(len(out)):
            out_index = np.zeros(MAX_DIMS, np.int32)
            a_index = np.zeros(MAX_DIMS, np.int32)

            count(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)

            for s in range(reduce_size):
                count(s, reduce_shape, a_index)
                for k in range(len(reduce_shape)):
                    if reduce_shape[k] != 1:
                        out_index[k] = a_index[k]
                j = index_to_position(out_index, a_strides)
                out[o] = fn(out[o], a[j])

    return njit(parallel=True)(_reduce)


def reduce(fn, start=0.0):
    f = tensor_reduce(njit()(fn))

    def ret(a, dims=None, out=None):
        if out is None:
            out_shape = list(a.shape)
            for d in dims:
                out_shape[d] = 1
            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

        diff = len(a.shape) - len(out.shape)

        reduce_shape = []
        reduce_size = 1
        for i, s in enumerate(a.shape):
            if i < diff or out.shape[i - diff] == 1:
                reduce_shape.append(s)
                reduce_size *= s
            else:
                reduce_shape.append(1)
        # assert len(out.shape) == len(a.shape)
        f(*out.tuple(), *a.tuple(), np.array(reduce_shape), reduce_size)
        return out

    return ret


class FastOps:
    map = map
    zip = zip
    reduce = reduce
