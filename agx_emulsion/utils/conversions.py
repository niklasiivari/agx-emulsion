import numpy as np
import colour
from opt_einsum import contract
import functools
from joblib import Parallel, delayed
import multiprocessing
import pyopencl as cl
import threading

from agx_emulsion.config import SPECTRAL_SHAPE
from agx_emulsion.utils.io import load_densitometer_data

# Global cache for OpenCL context and program.
_opencl_lock = threading.Lock()
_opencl_ctx = None
_opencl_program = None

def get_opencl_context_and_program(channels):
    global _opencl_ctx, _opencl_program
    with _opencl_lock:
        if _opencl_ctx is None:
            _opencl_ctx = cl.create_some_context()
        if _opencl_program is None:
            kernel_code = """
            __kernel void contract_kernel(
                __global const float *a,
                __global const float *M,
                __global float *result,
                const int channels)
            {
                int pixel = get_global_id(0);
                for (int l = 0; l < channels; l++) {
                    float sum = 0.0f;
                    for (int k = 0; k < channels; k++) {
                        sum += a[pixel * channels + k] * M[l * channels + k];
                    }
                    result[pixel * channels + l] = sum;
                }
            }
            """
            _opencl_program = cl.Program(_opencl_ctx, kernel_code).build()
    return _opencl_ctx, _opencl_program

def density_to_light(density, light):
    """
    Convert density to light transmittance.

    This function calculates the light transmittance based on the given density
    and light intensity. It uses the formula transmittance = 10^(-density) to 
    compute the transmittance and then multiplies it by the light intensity.

    Parameters:
    density (float or np.ndarray): The density value(s) which affect the light transmittance.
    light (float or np.ndarray): The initial light intensity value(s).

    Returns:
    np.ndarray: The light intensity after passing through the medium with the given density.
    """
    transmitted = 10**(-density) * light
    np.nan_to_num(transmitted, copy=False, nan=0.0)
    return transmitted


@functools.lru_cache(None)
def _cached_densitometer_data(type='status_A'):
    return load_densitometer_data(type=type)

def compute_densitometer_correction(dye_density, type='status_A'):
    densitometer_responsivities = _cached_densitometer_data(type=type)
    dye_density = dye_density[:,0:3]
    np.nan_to_num(dye_density, copy=False, nan=0.0)
    densitometer_correction = 1 / np.sum(densitometer_responsivities[:] * dye_density, axis=0)
    return densitometer_correction

@functools.lru_cache(maxsize=None)
def compute_aces_conversion_matrix(sensitivity, illuminant):
    msds = colour.MultiSpectralDistributions(sensitivity, domain=SPECTRAL_SHAPE.wavelengths)
    M, _ = colour.matrix_idt(msds, illuminant)
    return np.linalg.inv(M)

def parallel_contract(aces, aces_conversion_matrix):
    cores = multiprocessing.cpu_count()
    chunks = np.array_split(aces, cores, axis=0)
    results = Parallel(n_jobs=cores)(
        delayed(lambda a: contract('ijk,lk->ijl', a, aces_conversion_matrix))(chunk)
        for chunk in chunks
    )
    return np.concatenate(results, axis=0)

def opencl_parallel_contract(aces, aces_conversion_matrix):
    """
    GPU-accelerated contraction using cached OpenCL context.
    For each pixel, computes: result[i,j,l] = sum_k aces[i,j,k] * aces_conversion_matrix[l,k]
    """
    aces = np.array(aces, dtype=np.float32)
    M = np.array(aces_conversion_matrix, dtype=np.float32)
    height, width, channels = aces.shape
    ctx, program = get_opencl_context_and_program(channels)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=aces)
    M_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=M)
    result = np.empty((height * width * channels), dtype=np.float32)
    res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

    num_pixels = height * width
    program.contract_kernel(queue, (num_pixels,), None, a_buf, M_buf, res_buf, np.int32(channels))
    cl.enqueue_copy(queue, result, res_buf)
    return result.reshape((height, width, channels))

def rgb_to_raw_aces_idt(RGB, illuminant, sensitivity, midgray_rgb=[[[0.184,0.184,0.184]]],
                        color_space='sRGB', apply_cctf_decoding=True, aces_conversion_matrix=[],
                        use_gpu=True):
    """
    Converts RGB values to raw values using ACES IDT.
    If use_gpu is True, uses GPU acceleration via OpenCL.
    """
    aces = colour.RGB_to_RGB(RGB, color_space, 'ACES2065-1',
                    apply_cctf_decoding=apply_cctf_decoding,
                    apply_cctf_encoding=False)
    if aces_conversion_matrix == []:
        aces_conversion_matrix = compute_aces_conversion_matrix(
            tuple(map(tuple, sensitivity)),
            tuple(illuminant)
        )
    if use_gpu:
        raw = opencl_parallel_contract(aces, aces_conversion_matrix) / midgray_rgb
    else:
        raw = parallel_contract(aces, aces_conversion_matrix) / midgray_rgb
    raw_midgray = np.array([[[1,1,1]]])
    return raw, raw_midgray
