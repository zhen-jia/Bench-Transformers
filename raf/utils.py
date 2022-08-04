"""Utilities."""
from time import time
import subprocess

from logger import get_logger

logger = get_logger("Util")  # pylint: disable=invalid-name


def func_timer(prefix=None):
    """A decorator to time the execution time of a function."""

    def _do_timer(func):
        msg = func.__name__ if not prefix else prefix

        def _wrap_timer(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            end = time()
            logger.info("%s...%.2fs", msg, end - start)
            return result

        return _wrap_timer

    return _do_timer


def get_tensor_size(t_list):
    """Calculate the total tensor size in MBs in the given list.

    Parameters
    ----------
    t_list: Union[List[ndarray], List[torch.Tensor]]
        A list of tensors.

    Returns
    -------
    float:
        The total tensor size in MBs.
    """
    tensors_size = 0.0
    for tensor in t_list:
        dtype = str(tensor.dtype).split(".")[-1]
        if dtype in ["float32", "int32"]:
            nbytes = 4
        elif dtype in ["float64", "int64"]:
            nbytes = 8
        elif dtype in ["float16", "int16"]:
            nbytes = 2
        elif dtype in ["int8", "uint8"]:
            nbytes = 1
        else:
            raise RuntimeError("Not support date type %s" % tensor.dtype)

        nsize = 1
        for shape in tensor.shape:
            nsize *= shape
        tensors_size += nsize * nbytes / 1048576.0

    return tensors_size


def get_gpu_total_memory():
    """Get the total memory of the GPU.

    Returns
    -------
    memory: int
        The total memory of the GPU in MiB.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        check=True,
    )
    return int(result.stdout)


def get_gpu_free_memory():
    """Get the free memory of the GPU.

    Returns
    -------
    memory: int
        The total memory of the GPU in MiB.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        check=True,
    )
    return int(result.stdout)
