import inspect
import numpy as np
import warnings

from typing import List


def calc_batch_size(unit_flop, mem_avail, pre_flop=0, dtype=float, min_batch=1):
    """ Calculate batch size within possible memory.

    For example, if we want to compute tensor (100, 100, 100), but only 50,000 memory available,
    then this tensor should be splited into 20 batches.

    ``flop`` in parameters is number of data, not refers to FLOPs.

    Parameters
    ----------
    unit_flop : int
        Number of data for unit operation.

        For example, for a tensor with shape (110, 120, 130), the 1st dimension is indexable from
        outer programs, then a unit operation handles 120x130 = 15,600 data. Then we call this function
        with ``unit_flop = 15600``.

        This value will be set to 1 if too small.
    mem_avail : float
        Memory available in MB.
    pre_flop : int
        Number of data preserved in memory. Unit in number.
    dtype : type
        Type of data. Such as np.float64, complex, etc.
    min_batch : int
        Minimum value of batch.

        If this value set to 0 (by default), then when memory overflow
        detected, an exception will be raised. Otherwise, only a warning
        will be raised and try to fill memory as possible.

    Returns
    -------
    batch_size : int
        Size of one batch available for outer function iteration.
    """
    unit_flop = max(unit_flop, 1)
    unit_mb = unit_flop * np.dtype(dtype).itemsize / 1024**2
    max_mb = mem_avail - pre_flop * np.dtype(dtype).itemsize / 1024 ** 2

    if unit_mb * max(min_batch, 1) > max_mb:
        warning_token = "Memory overflow when preparing batch number. " \
                        "Current memory available {:10.3f} MB, minimum required {:10.3f} MB." \
                        .format(max_mb, unit_mb * max(min_batch, 1))
        if min_batch <= 0:
            raise ValueError(warning_token)
        else:
            warnings.warn(warning_token)

    batch_size = int(max(max_mb / unit_mb, unit_mb))
    batch_size = max(batch_size, min_batch)
    return batch_size


def parse_incore_flag(flag, unit_flop, mem_avail, pre_flop=0, dtype=float):
    """ Parse flag of whether tensor can be stored incore.

    ``flop`` in parameters is number of data, not refers to FLOPs.

    Parameters
    ----------
    flag : bool or float or None or str
        Input flag.

        - True: Store tensor in memory.
        - False: Store tensor in disk.
        - None: Store tensor nowhere.
        - "auto": Judge tensor in memory/disk by available memory.
        - (float): Judge tensor in memory/disk by given value in MB.
    unit_flop : int
        Number of data for unit operation.

        For example, for a tensor with shape (110, 120, 130), the 1st dimension is indexable from
        outer programs, then a unit operation handles 120x130 = 15,600 data. Then we call this function
        with ``unit_flop = 15600``.

        This value will be set to 1 if too small.
    mem_avail : float
        Memory available in MB.
    pre_flop : int
        Number of data preserved in memory. Unit in number.
    dtype : type
        Type of data. Such as np.float64, complex, etc.

    Returns
    -------
    True or False or None
        Output flag of whether tensor store in memory/disk/nowhere.
    """
    if flag in [False, True, None]:
        return flag
    if isinstance(flag, str) and flag.lower().strip() == "auto":
        pass
    else:  # assert flag is a number
        mem_avail = float(flag)
    unit_flop = max(unit_flop, 1)
    unit_mb = unit_flop * np.dtype(dtype).itemsize / 1024**2
    max_mb = mem_avail - pre_flop * np.dtype(dtype).itemsize / 1024 ** 2
    return unit_mb < max_mb


def gen_batch(val_min, val_max, batch_size):
    """ Generate slices given numbers of batch.

    Parameters
    ----------
    val_min : int
        Minimum value to be iterated
    val_max : int
        Maximum value to be iterated
    batch_size : int
        Batch size to be sliced.

    Returns
    -------
    List[slice]

    Examples
    --------
    >>> gen_batch(10, 20, 3)
        [slice(10, 13, None), slice(13, 16, None), slice(16, 19, None), slice(19, 20, None)]
    """
    return [slice(i, (i + batch_size) if i + batch_size < val_max else val_max)
            for i in range(val_min, val_max, batch_size)]


def gen_leggauss_0_inf(ngrid):
    x, w = np.polynomial.legendre.leggauss(ngrid)
    return 0.5 * (1 + x) / (1 - x), w / (1 - x)**2


def gen_leggauss_0_1(ngrid):
    x, w = np.polynomial.legendre.leggauss(ngrid)
    return 0.5 * (x + 1), 0.5 * w


def check_real(var, rtol=1e-5, atol=1e-8):
    """ Check and return array or complex number is real.

    Parameters
    ----------
    var : complex or np.ndarray
        Complex value to be checked.
    rtol : float
        Relative error threshold.
    atol : float
        Absolute error threshold.

    Returns
    -------
    complex or np.ndarray
    """
    if not np.allclose(np.real(var), var, rtol=rtol, atol=atol):
        caller_locals = inspect.currentframe().f_back.f_locals
        for key, val in caller_locals.items():
            if id(var) == id(val):
                raise ValueError("Variable `{:}` is not real.".format(key))
    else:
        return np.real(var)


def sanity_dimension(array, shape, weak=False):
    """ Sanity check for array dimension.

    Parameters
    ----------
    array : np.ndarray
        The data to be checked. Should have attribute ``shape``.
    shape : tuple[int]
        Shape of data to be checked.
    weak : bool
        If weak, then only check size of array; otherwise, check dimension
        shape. Default to False.
    """
    caller_locals = inspect.currentframe().f_back.f_locals
    for key, val in caller_locals.items():
        if id(array) == id(val):
            if not weak:
                if array.shape != shape:
                    raise ValueError(
                        "Dimension sanity check: {:} is not {:}"
                        .format(key, shape))
            else:
                if np.prod(array.shape) != np.prod(shape):
                    pass
                raise ValueError(
                    "Dimension sanity check: Size of {:} is not {:}"
                    .format(np.prod(array.shape), np.prod(array.shape)))
            return
    raise ValueError("Array in dimension sanity check does not included in "
                     "upper caller function.")


def pad_omega(s, omega):
    """ Pad omega parameter ``_omega({:.6f})`` after string if RSH parameter omega is not zero.

    Padding always returns 6 float digitals.

    Parameters
    ----------
    s : str
    omega : float

    Returns
    -------
    str
    """
    if omega == 0:
        return s
    return f"{s}_omega({omega:.6f})"


def allocate_array(incore, shape, max_memory,
                   h5file=None, name=None, dtype=float, zero_init=True, chunks=None, **kwargs):
    """ Allocate an array with given memory estimation and incore stragety.

    Parameters
    ----------
    incore : bool or float or None or str
        Incore flag. Also see :class:`parse_incore_flag`.
    shape : tuple
        Shape of allocated array.
    max_memory : int or float
        Maximum memory in MB.
    h5file : h5py.File
        HDF5 file instance. Array may save into this file if disk is required.
    name : str
        Name of array; Only useful when h5py-based.
    dtype : np.dtype
        Type of numpy array.
    zero_init : bool
        Initialize array as zero if required; only useful when numpy-based.
    chunks : tuple
        Chunk of array; only useful when h5py-based.

    Returns
    -------
    np.array or h5py.Dataset
    """
    incore = parse_incore_flag(incore, int(np.prod(shape)), max_memory, dtype=dtype)
    if incore is None:
        return None
    elif incore is True:
        if zero_init:
            return np.zeros(shape, dtype=dtype)
        else:
            return np.empty(shape, dtype=dtype)
    else:
        assert incore is False
        if h5file is None:
            # this line of code require calling pyscf
            from pyscf import lib
            h5file = lib.misc.H5TmpFile()
        if name is None:
            import string
            import random
            name = "".join(random.choices(string.ascii_letters, k=6))
            while name not in h5file:
                name = "".join(random.choices(string.ascii_letters, k=6))
        return h5file.create_dataset(name=name, shape=shape, chunks=chunks, dtype=dtype, **kwargs)


def update_results(results, income_result, allow_overwrite=True, warn_overwrite=True):
    """ Update results as dictionary with warnings.

    This function may change attribute ``results``.

    Parameters
    ----------
    results : dict
        Result dictionary to be updated.
    income_result : dict
        Result dictionary to be merged into ``results``.
    allow_overwrite : bool
        Whether allows overwriting result dictionary.
    warn_overwrite : bool
        Whether warns overwriting result dictionary.
    """
    if not allow_overwrite or warn_overwrite:
        keys_interset = set(income_result).intersection(results)
        if len(keys_interset) != 0:
            if not allow_overwrite:
                raise KeyError(f"Overwrite results is not allowed!\nRepeated keys: {keys_interset}")
            if warn_overwrite:
                msg = "Key result overwrited!\n"
                for key in keys_interset:
                    # if results are very close, then warn muted; otherwise, it could be very annoying
                    try:
                        if np.allclose(income_result[key], results[key], atol=1e-10, rtol=1e-10):
                            continue
                    except TypeError:
                        pass
                    msg += f"Key: {key}, before {income_result[key]}, after {results[key]}\n"
                    warnings.warn(msg)
    results.update(income_result)
    return results
