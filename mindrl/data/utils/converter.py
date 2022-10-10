import pickle
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Optional, Union, no_type_check

import h5py
import numpy as np
import mindspore as ms
from mindspore import ops

from mindrl.data.batch import Batch, _parse_value


@no_type_check
def to_numpy(x: Any) -> Union[Batch, np.ndarray]:
    """Return an object without torch.Tensor."""
    if isinstance(x, ms.Tensor):  # most often case
        return x.asnumpy()
    elif isinstance(x, np.ndarray):  # second often case
        return x
    elif isinstance(x, (np.number, np.bool_, Number)):
        return np.asanyarray(x)
    elif x is None:
        return np.array(None, dtype=object)
    elif isinstance(x, (dict, Batch)):
        x = Batch(x) if isinstance(x, dict) else deepcopy(x)
        x.to_numpy()
        return x
    elif isinstance(x, (list, tuple)):
        return to_numpy(_parse_value(x))
    else:  # fallback
        return np.asanyarray(x)


@no_type_check
def to_mindspore(
    x: Any,
    dtype: ms.dtype = None,
    device: Union[str, int] = "CPU",
) -> Union[Batch, ms.Tensor]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(
        x.dtype.type, (np.bool_, np.number)
    ):  # most often case
        x = ms.Tensor.from_numpy(x)
        if dtype is not None:
            x = ms.Tensor(x,dtype=dtype)
        return x
    elif isinstance(x, ms.Tensor):  # second often case
        if dtype is not None:
            x = ms.Tensor(x,dtype=dtype)
        return x
    elif isinstance(x, (np.number, np.bool_, Number)):
        return to_mindspore(np.asanyarray(x), dtype, device)
    elif isinstance(x, (dict, Batch)):
        x = Batch(x, copy=True) if isinstance(x, dict) else deepcopy(x)
        x.to_mindspore(dtype, device)
        return x
    elif isinstance(x, (list, tuple)):
        return to_mindspore(_parse_value(x), dtype)
    else:  # fallback
        raise TypeError(f"object {x} cannot be converted to torch.")


@no_type_check
def to_mindspore_as(x: Any, y: ms.Tensor) -> Union[Batch, ms.Tensor]:
    """Return an object without np.ndarray.

    Same as ``to_torch(x, dtype=y.dtype, device=y.device)``.
    """
    assert isinstance(y, ms.Tensor)
    return to_mindspore(x, dtype=y.dtype)


# Note: object is used as a proxy for objects that can be pickled
# Note: mypy does not support cyclic definition currently
Hdf5ConvertibleValues = Union[  # type: ignore
    int, float, Batch, np.ndarray, ms.Tensor, object,
    'Hdf5ConvertibleType',  # type: ignore
]

Hdf5ConvertibleType = Dict[str, Hdf5ConvertibleValues]  # type: ignore


def to_hdf5(
    x: Hdf5ConvertibleType, y: h5py.Group, compression: Optional[str] = None
) -> None:
    """Copy object into HDF5 group."""

    def to_hdf5_via_pickle(
        x: object, y: h5py.Group, key: str, compression: Optional[str] = None
    ) -> None:
        """Pickle, convert to numpy array and write to HDF5 dataset."""
        data = np.frombuffer(pickle.dumps(x), dtype=np.byte)
        y.create_dataset(key, data=data, compression=compression)

    for k, v in x.items():
        if isinstance(v, (Batch, dict)):
            # dicts and batches are both represented by groups
            subgrp = y.create_group(k)
            if isinstance(v, Batch):
                subgrp_data = v.__getstate__()
                subgrp.attrs["__data_type__"] = "Batch"
            else:
                subgrp_data = v
            to_hdf5(subgrp_data, subgrp, compression=compression)
        elif isinstance(v, ms.Tensor):
            # MindSpore tensors are written to datasets
            y.create_dataset(k, data=to_numpy(v), compression=compression)
            y[k].attrs["__data_type__"] = "Tensor"
        elif isinstance(v, np.ndarray):
            try:
                # NumPy arrays are written to datasets
                y.create_dataset(k, data=v, compression=compression)
                y[k].attrs["__data_type__"] = "ndarray"
            except TypeError:
                # If data type is not supported by HDF5 fall back to pickle.
                # This happens if dtype=object (e.g. due to entries being None)
                # and possibly in other cases like structured arrays.
                try:
                    to_hdf5_via_pickle(v, y, k, compression=compression)
                except Exception as exception:
                    raise RuntimeError(
                        f"Attempted to pickle {v.__class__.__name__} due to "
                        "data type not supported by HDF5 and failed."
                    ) from exception
                y[k].attrs["__data_type__"] = "pickled_ndarray"
        elif isinstance(v, (int, float)):
            # ints and floats are stored as attributes of groups
            y.attrs[k] = v
        else:  # resort to pickle for any other type of object
            try:
                to_hdf5_via_pickle(v, y, k, compression=compression)
            except Exception as exception:
                raise NotImplementedError(
                    f"No conversion to HDF5 for object of type '{type(v)}' "
                    "implemented and fallback to pickle failed."
                ) from exception
            y[k].attrs["__data_type__"] = v.__class__.__name__


def from_hdf5(x: h5py.Group, device: Optional[str] = None) -> Hdf5ConvertibleValues:
    """Restore object from HDF5 group."""
    if isinstance(x, h5py.Dataset):
        # handle datasets
        if x.attrs["__data_type__"] == "ndarray":
            return np.array(x)
        elif x.attrs["__data_type__"] == "Tensor":
            return ms.Tensor(x)
        else:
            return pickle.loads(x[()])
    else:
        # handle groups representing a dict or a Batch
        y = dict(x.attrs.items())
        data_type = y.pop("__data_type__", None)
        for k, v in x.items():
            y[k] = from_hdf5(v, device)
        return Batch(y) if data_type == "Batch" else y
