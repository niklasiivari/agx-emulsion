import numpy as np
import colour

# Constants
ENLARGER_STEPS = 170
LOG_EXPOSURE = np.linspace(-3,4,256)
SPECTRAL_SHAPE = colour.SpectralShape(380, 780, 5)

# Default color matching functions
STANDARD_OBSERVER_CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(SPECTRAL_SHAPE)

USE_OPENCL = True
# New toggles to control individual OpenCL accelerated methods
USE_OPENCL_CONTRACT = True      # Toggle for OpenCL contract calculations
USE_OPENCL_BLUR = True          # Toggle for OpenCL blur operations
USE_OPENCL_LUT_CUBIC = True     # Toggle for OpenCL cubic LUT interpolation
USE_OPENCL_RESIZE = True    # Toggle for OpenCL linear LUT interpolation
USE_OPENCL_LUT3D = True         # Toggle for OpenCL 3D LUT interpolation