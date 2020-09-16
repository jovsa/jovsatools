import numpy as np
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)


def tensor_map(fn):
    """
    Higher-order tensor map function.

    Args:
        fn: function from float-to-float to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
       None : Fills in `out`.
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        # ASSIGN2.2
        out_index = np.zeros(MAX_DIMS, np.int32)
        in_index = np.zeros(MAX_DIMS, np.int32)
        for i in range(len(out)):
            count(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])
        # END ASSIGN2.2

    return _map


def map(fn):
    """
    Higher-order tensor map function.

    Args:
        fn: function from float-to-float to apply.
        a (:class:`TensorData`): tensor to map over
        out (:class:`TensorData`): optional, tensor data to fill in,
        should broadcast with `a`.
    Returns:
       :class:`TensorData` : new tensor data
    """

    f = tensor_map(fn)

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
        fn: function mapping two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
       None : Fills in `out`.
    """

    def _zip(out, out_shape, out_strides, a, a_shape, a_strides, b, b_shape, b_strides):
        # ASSIGN2.2
        out_index = np.zeros(MAX_DIMS, np.int32)
        a_index = np.zeros(MAX_DIMS, np.int32)
        b_index = np.zeros(MAX_DIMS, np.int32)
        for i in range(len(out)):
            count(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a[j], b[k])
        # END ASSIGN2.2

    return _zip


def zip(fn):
    """
    Higher-order tensor zip function.

    Args:
        fn: function from two floats-to-float to apply.
        a (:class:`TensorData`): tensor to zip over
        b (:class:`TensorData`): tensor to zip over
    Returns:
       :class:`TensorData` : new tensor data
    """

    f = tensor_zip(fn)

    def ret(a, b):
        if a.shape != b.shape:
            c_shape = shape_broadcast(a.shape, b.shape)
        else:
            c_shape = a.shape
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

    Returns:
       None : Fills in `out`.
    """

    def _reduce(
        out, out_shape, out_strides, a, a_shape, a_strides, reduce_shape, reduce_size
    ):
        # ASSIGN2.2
        out_index = np.zeros(MAX_DIMS, np.int32)
        a_index = np.zeros(MAX_DIMS, np.int32)
        for i in range(len(out)):
            count(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)

            for s in range(reduce_size):
                count(s, reduce_shape, a_index)
                for i in range(len(reduce_shape)):
                    if reduce_shape[i] != 1:
                        out_index[i] = a_index[i]
                j = index_to_position(out_index, a_strides)
                out[o] = fn(out[o], a[j])
        # END ASSIGN2.2

    return _reduce


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function.

    Args:
        fn: function from two floats-to-float to apply.
        a (:class:`TensorData`): tensor to reduce over
        dims (list, optional): list of dims to reduce
        out (:class:`TensorData`, optional): tensor to reduce into

    Returns:
       :class:`TensorData` : new tensor data
    """

    f = tensor_reduce(fn)

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
        assert len(out.shape) == len(a.shape)
        f(*out.tuple(), *a.tuple(), reduce_shape, reduce_size)
        return out

    return ret


class TensorOps:
    map = map
    zip = zip
    reduce = reduce
