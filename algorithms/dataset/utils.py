import numpy as np
import collections
import h5py


def pad_tensor(x, max_len, axis):
    pad_widths = [(0,0) for _ in range(len(x.shape))]
    pad_widths[axis] = (0, max_len - x.shape[axis])
    return np.pad(x, (pad_widths), mode='constant')


def compute_n_batches(n_samples, batch_size):
    n_batches = int(n_samples / batch_size)
    if n_samples % batch_size != 0:
        n_batches += 1
    return n_batches


class KeyValueReplayMemory(object):

    def __init__(self, maxsize=None):
        self.maxsize = maxsize
        self.mem = collections.defaultdict(list)

    def add(self, keys, values):
        '''
        Adds keys from values to memory
        Args:
            - keys: the keys to add, list of hashable
            - values: dict containing each key in keys
        '''
        n_samples = len(values[keys[0]])
        for key in keys:
            assert len(values[key]) == n_samples, 'n_samples from each key must match'
            self.mem[key].extend(values[key])
            if self.maxsize:
                self.mem[key] = self.mem[key][-self.maxsize:]

    def sample(self, keys, size):
        '''
        Sample a batch of size for each key and return as a dict
        Args:
            - keys: list of keys
            - size: number of samples to select
        '''
        sample = dict()
        n_samples = len(self.mem[keys[0]])
        idxs = np.random.randint(0, n_samples, size)
        for key in keys:
            sample[key] = np.take(self.mem[key], idxs, axis=0)
        return sample


def load_dataset(filepath, maxsize=None):
    f = h5py.File(filepath, 'r')
    d = dict()
    for key in f.keys():
        if maxsize is None:
            d[key] = f[key].value
        else:
            d[key] = f[key].value[:maxsize]
    return d

