import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


a=1

def fn(x, y, z):
    if a==1:
        res = x * ops.exp(y) * ops.pow(z, 2)
    else:
        res = x * ops.exp(y) + ops.pow(z, 2)
    return res, z


if __name__=="__main__":
    x = np.array([3, 3]).astype(np.float)
    y = np.array([0, 0]).astype(np.float)
    z = np.array([5, 5]).astype(np.float)

    x = ms.Tensor.from_numpy(x)
    y = ms.Tensor.from_numpy(y)
    z = ms.Tensor.from_numpy(z)

    output, gradient = ops.value_and_grad(fn, grad_position=(1, 2), weights=None, has_aux=True)(x, y, z)

    print(output)

    print(gradient)
