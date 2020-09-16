import math

## Task 0.1
## Mathematical operators


def mul(x, y):
    ":math:`f(x, y) = x * y`"
    # ASSIGN0.1
    return x * y
    # END ASSIGN0.1


def id(x):
    ":math:`f(x) = x`"
    # ASSIGN0.1
    return x
    # END ASSIGN0.1


def add(x, y):
    ":math:`f(x, y) = x + y`"
    # ASSIGN0.1
    return x + y
    # END ASSIGN0.1


def neg(x):
    ":math:`f(x) = -x`"
    # ASSIGN0.1
    return -x
    # END ASSIGN0.1


def lt(x, y):
    ":math:`f(x) =` 1.0 if x is less than y else 0.0"
    # ASSIGN0.1
    return 1.0 if x < y else 0.0
    # END ASSIGN0.1


def eq(x, y):
    ":math:`f(x) =` 1.0 if x is equal to y else 0.0"
    # ASSIGN0.1
    return 1.0 if x == y else 0.0
    # END ASSIGN0.1


def max(x, y):
    ":math:`f(x) =` x if x is greater than y else y"
    # ASSIGN0.1
    return x if x > y else y
    # END ASSIGN0.1


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    """
    # ASSIGN0.1
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
    # END ASSIGN0.1


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)
    """
    # ASSIGN0.1
    return x if x > 0 else 0.0
    # END ASSIGN0.1


def relu_back(x, y):
    ":math:`f(x) =` y if x is greater than 0 else 0"
    # ASSIGN0.1
    return y if x > 0 else 0.0
    # END ASSIGN0.1


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    return math.log(x + EPS)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(a, b):
    return b / (a + EPS)


def inv(x):
    ":math:`f(x) = 1/x`"
    return 1.0 / x


def inv_back(a, b):
    return -(1.0 / a ** 2) * b


## Task 0.3
## Higher-order functions.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    # ASSIGN0.3
    def _map(ls):
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map
    # END ASSIGN0.3


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) one each pair of elements.

    """
    # ASSIGN0.3
    def _zipWith(ls1, ls2):
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith
    # END ASSIGN0.3


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(fn, start):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`

    """
    # ASSIGN0.3
    def _reduce(ls):
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce
    # END ASSIGN0.3


def sum(ls):
    """
    Sum up a list using :func:`reduce` and :func:`add`.
    """
    # ASSIGN0.3
    return reduce(add, 0.0)(ls)
    # END ASSIGN0.3


def prod(ls):
    """
    Product of a list using :func:`reduce` and :func:`mul`.
    """
    # ASSIGN0.3
    return reduce(mul, 1.0)(ls)
    # END ASSIGN0.3
