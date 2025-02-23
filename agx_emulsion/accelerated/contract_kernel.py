# Updated contract_kernel.py
import pyopencl as cl
from agx_emulsion.accelerated.opencl_context import get_context

KERNEL_CODE = """
__kernel void contract_kernel(
    __global const float *a,
    __global const float *M,
    __global float *result,
    const int input_channels,
    const int output_channels)
{
    int pixel = get_global_id(0);
    for (int l = 0; l < output_channels; l++) {
        float sum = 0.0f;
        for (int k = 0; k < input_channels; k++) {
            sum += a[pixel * input_channels + k] * M[l * input_channels + k];
        }
        result[pixel * output_channels + l] = sum; // Removed clamp for correctness
    }
}
"""

KERNEL_CODE_ijkl = """
__kernel void contract_kernel_ijkl(
    __global const float *a,
    __global const float *M,
    __global float *result,
    const int L,
    const int M_dim)
{
    int pixel = get_global_id(0);
    for (int m = 0; m < M_dim; m++) {
        float sum = 0.0f;
        for (int l = 0; l < L; l++) {
            sum += a[pixel * L + l] * M[l * M_dim + m];
        }
        result[pixel * M_dim + m] = sum;
    }
}
"""

def get_contract_kernel():
    ctx = get_context()
    program = cl.Program(ctx, KERNEL_CODE).build()
    return program.contract_kernel

def get_contract_kernel_ijkl():
    ctx = get_context()
    program = cl.Program(ctx, KERNEL_CODE_ijkl).build()
    return program.contract_kernel_ijkl