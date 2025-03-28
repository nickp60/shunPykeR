import numpy
import ctypes

dll = ctypes.CDLL(numpy.core._multiarray_umath.__file__)
get_config = dll.openblas_get_config
get_config.restype=ctypes.c_char_p
res = get_config()
print('OpenBLAS get_config returned', str(res))
