import minitorch
from hypothesis import given
from .strategies import tensors, assert_close
import pytest


@pytest.mark.task4_2
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t):
    out = minitorch.avgpool2d(t, (2, 2))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_2
@given(tensors(shape=(1, 1, 4, 4)))
def test_max(t):
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t):
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0


@pytest.mark.task4_1
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t):
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 6, 6)), tensors(shape=(1, 1, 2, 3)))
def test_conv(input, weight):
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_3
@given(tensors(shape=(2, 1, 6, 6)), tensors(shape=(1, 1, 2, 3)))
def test_conv_batch(input, weight):
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_3
@given(tensors(shape=(2, 2, 6, 6)), tensors(shape=(3, 2, 2, 3)))
def test_conv_channel(input, weight):
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_3
def test_conv2():
    t = minitorch.tensor_fromlist(
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    ).view(1, 1, 4, 4)
    t.requires_grad_(True)

    t2 = minitorch.tensor_fromlist([[1, 1], [1, 1]]).view(1, 1, 2, 2)
    t2.requires_grad_(True)
    out = minitorch.Conv2dFun.apply(t, t2)
    out.sum().backward()

    minitorch.grad_check(minitorch.Conv2dFun.apply, t, t2)
