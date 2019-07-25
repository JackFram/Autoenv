import numpy as np


# Immutable, lazily evaluated dict
class lazydict(object):
    def __init__(self, **kwargs):
        self._lazy_dict = kwargs
        self._dict = {}

    def __getitem__(self, key):
        if key not in self._dict:
            self._dict[key] = self._lazy_dict[key]()
        return self._dict[key]

    def __setitem__(self, i, y):
        self.set(i, y)

    def get(self, key, default=None):
        if key in self._lazy_dict:
            return self[key]
        return default

    def set(self, key, value):
        self._lazy_dict[key] = value


"""
Devide function f's inputs into several slices. Evaluate f on those slices, and then average the result. It is useful 
when memory is not enough to process all data at once.
Assume:
1. each of f's inputs is iterable and composed of multiple "samples"
2. outputs can be averaged over "samples"
"""


def sliced_fun(f, n_slices):
    def sliced_f(sliced_inputs, non_sliced_inputs=None):
        if non_sliced_inputs is None:
            non_sliced_inputs = []
        if isinstance(non_sliced_inputs, tuple):
            non_sliced_inputs = list(non_sliced_inputs)
        n_paths = len(sliced_inputs[0])
        slice_size = max(1, n_paths // n_slices)
        ret_vals = None
        for start in range(0, n_paths, slice_size):
            inputs_slice = [v[start:start + slice_size] for v in sliced_inputs]
            slice_ret_vals = f(*(inputs_slice + non_sliced_inputs))
            if not isinstance(slice_ret_vals, (tuple, list)):
                slice_ret_vals_as_list = [slice_ret_vals]
            else:
                slice_ret_vals_as_list = slice_ret_vals
            scaled_ret_vals = [
                np.asarray(v) * len(inputs_slice[0]) for v in slice_ret_vals_as_list]
            if ret_vals is None:
                ret_vals = scaled_ret_vals
            else:
                ret_vals = [x + y for x, y in zip(ret_vals, scaled_ret_vals)]
        ret_vals = [v / n_paths for v in ret_vals]
        if not isinstance(slice_ret_vals, (tuple, list)):
            ret_vals = ret_vals[0]
        elif isinstance(slice_ret_vals, tuple):
            ret_vals = tuple(ret_vals)
        return ret_vals

    return sliced_f


def extract(x, *keys):
    if isinstance(x, (dict, lazydict)):
        return tuple(x[k] for k in keys)
    elif isinstance(x, list):
        return tuple([xi[k] for xi in x] for k in keys)
    else:
        raise NotImplementedError
