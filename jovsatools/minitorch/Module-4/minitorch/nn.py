import numpy as np
from .fast_ops import FastOps
from .tensor import rand, Function
from . import operators
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    MAX_DIMS,
)
from numba import njit, prange


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input, dim):
    """
    Compute the argmax as a 1-hot tensor.
    Args:
       input (:class:`Tensor`): input tensor
       dim (int): dimension to apply argmax
    Returns:
       :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise
    """
    out = max_reduce(input, [dim])
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        "Forward of max should be max reduction"
        out = max_reduce(input, [dim])
        ctx.save_for_backward(input, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"
        input, out = ctx.saved_values
        return (out == input) * grad_output, None


max = Max.apply


def softmax(input, dim):
    r"""
    Compute the softmax as a tensor.
    .. math::
        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}
    Args:
       input (:class:`Tensor`): input tensor
       dim (int): dimension to apply argmax
    Returns:
       :class:`Tensor` : softmax tensor
    """
    e = input.exp()
    partition = e.sum(dim=dim)
    return e / partition


def logsoftmax(input, dim):
    r"""
    Compute the log of the softmax as a tensor.
    .. math::
        z_i = x_i - \log \sum_i e^{x_i}
    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    Args:
       input (:class:`Tensor`): input tensor
       dim (int): dimension to apply argmax
    Returns:
       :class:`Tensor` : log of softmax tensor
    """
    e = input
    mx = max(e, dim)
    lse = (e - mx).exp().sum(dim=dim).log() + mx
    return e - lse


def tile(input, kernel):
    """
    Reshape an image tensor for 2D pooling
    Args:
       input (:class:`Tensor`): batch x channel x height x width
       kernel ((int, int)): height x width of pooling
    Returns:
       (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x kernel_height x kernel_width as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_width = width // kw
    new_height = height // kh

    x = input.view(batch, channel, new_height, kh, new_width, kw)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
    x = x.view(batch, channel, new_height, new_width, kh * kw)
    return x, new_height, new_width


def maxpool2d(input, kernel):
    """
    Tiled max pooling 2D
    Args:
       input (:class:`Tensor`): batch x channel x height x width
       kernel ((int, int)): height x width of pooling
    Returns:
       :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    x, new_height, new_width = tile(input, kernel)
    return max(x, 4).view(batch, channel, new_height, new_width)


def avgpool2d(input, kernel):
    """
    Tiled average pooling 2D
    Args:
       input (:class:`Tensor`): batch x channel x height x width
       kernel ((int, int)): height x width of pooling
    Returns:
       :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    x, new_height, new_width = tile(input, kernel)
    return x.mean(dim=4).view(batch, channel, new_height, new_width)


count = njit()(count)
index_to_position = njit()(index_to_position)
broadcast_index = njit()(broadcast_index)


@njit(parallel=True)
def tensor_conv2d(
    output,
    output_shape,
    output_strides,
    out_size,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    2D Convolution implementation.
    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): Compute forward (False) or backward conv
    """
    batch, in_channels, height, width = input_shape
    _, _, kh, kw = weight_shape

    for i in prange(out_size):
        out_index = np.zeros(MAX_DIMS, np.int32)
        count(i, output_shape, out_index)
        b = out_index[0]
        oc = out_index[1]
        h = out_index[2]
        w = out_index[3]
        for dh in range(kh):
            for dw in range(kw):
                ih, iw = h + dh, w + dw
                if reverse:
                    ih, iw = h - dh, w - dw
                if ih < 0 or ih >= height or iw < 0 or iw >= width:
                    continue
                for ic in range(in_channels):
                    s1 = input_strides
                    term1 = input[s1[0] * b + s1[1] * ic + s1[2] * ih + s1[3] * iw]
                    s2 = weight_strides
                    term2 = weight[s2[0] * oc + s2[1] * ic + s2[2] * dh + s2[3] * dw]
                    output[i] += term1 * term2


@njit(parallel=True)
def _conv2d_back_weight(
    grad_output,
    grad_output_shape,
    grad_output_strides,
    input,
    input_shape,
    input_strides,
    grad_weight,
    grad_weight_shape,
    grad_weight_strides,
    grad_weight_size,
):
    batch, in_channels, height, width = input_shape
    for i in prange(grad_weight_size):
        grad_weight_index = np.zeros(MAX_DIMS, np.int32)
        count(i, grad_weight_shape, grad_weight_index)
        oc = grad_weight_index[0]
        ic = grad_weight_index[1]
        dh = grad_weight_index[2]
        dw = grad_weight_index[3]
        for h in range(height):
            for w in range(width):
                ih, iw = h - dh, w - dw
                if ih < 0 or ih >= height or iw < 0 or iw >= width:
                    continue
                for b in range(batch):
                    s1 = input_strides
                    term1 = input[s1[0] * b + s1[1] * ic + s1[2] * h + s1[3] * w]
                    s2 = grad_output_strides
                    term2 = grad_output[
                        s2[0] * b + s2[1] * oc + s2[2] * ih + s2[3] * iw
                    ]
                    grad_weight[i] += term1 * term2


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Args:
            input (:class:`tensor`) : batch x in_channel x h x w
            weight (:class:`tensor`) : out_channel x in_channel x kh x kw
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        out_channels, in_channels, kh, kw = weight.shape
        grad_weight = grad_output.zeros(weight.shape)
        _conv2d_back_weight(
            *grad_output.tuple(),
            *input.tuple(),
            *grad_weight.tuple(),
            grad_weight.size,
        )
        grad_input = grad_output.zeros(input.shape)
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply


def dropout(input, rate, ignore=False):
    """
    Dropout dimensions based on random noise
    Args:
       input (:class:`Tensor`): input tensor
       rate (float): probability of dropping out each dimension
       ignore (bool): skip
    Returns:
       :class:`Tensor` : tensor with dropout dimensions
    """
    if ignore:
        return input
    r = rand(input.shape)
    drop = rate < r
    return input * drop