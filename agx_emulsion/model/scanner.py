import numpy as np
import colour
from opt_einsum import contract

from agx_emulsion import config
from agx_emulsion.accelerated.opencl_accelerated import apply_gaussian_blur_gpu, apply_combined_gaussian_unsharp_gpu

from agx_emulsion.model.diffusion import apply_unsharp_mask, apply_gaussian_blur

from agx_emulsion.utils.conversions import density_to_light

from agx_emulsion.accelerated.gpu_contract import opencl_parallel_contract

from agx_emulsion.model.emulsion import compute_random_glare_amount

def scan(self,
            density_spectral,
            illuminant,
            color_space='sRGB',
            apply_cctf_encoding=True,
            lens_blur=0.0,
            unsharp_mask=[0.0,0.8]):
    light = self._calculate_light_transmitted(density_spectral, illuminant)
    rgb   = self._add_glare_and_convert_light_to_RGB(light, illuminant, color_space)
    rgb   = self._apply_blur_and_unsharp(rgb, lens_blur, unsharp_mask)
    rgb   = self._apply_cctf_encoding_and_clip(rgb, color_space, apply_cctf_encoding)
    return rgb
    
def _calculate_light_transmitted(self, density_spectral, illuminant):
    return density_to_light(density_spectral, illuminant)

def _add_glare_and_convert_light_to_RGB(self, light_transmitted, illuminant, color_space):
    from agx_emulsion import config
    normalization = np.sum(illuminant * config.STANDARD_OBSERVER_CMFS[:, 1], axis=0)
    if config.USE_OPENCL_CONTRACT:
        xyz = opencl_parallel_contract('ijk,kl->ijl', light_transmitted, config.STANDARD_OBSERVER_CMFS[:]) / normalization
        illuminant_xyz = opencl_parallel_contract('k,kl->l', illuminant, config.STANDARD_OBSERVER_CMFS[:]) / normalization
    else:
        xyz = contract('ijk,kl->ijl', light_transmitted, config.STANDARD_OBSERVER_CMFS[:]) / normalization
        illuminant_xyz = contract('k,kl->l', illuminant, config.STANDARD_OBSERVER_CMFS[:]) / normalization
    if self.type=='paper' and self.glare.active and self.glare.percent>0:
        glare_amount = compute_random_glare_amount(self.glare.percent, self.glare.roughness, self.glare.blur, light_transmitted.shape[:2])
        xyz += glare_amount[:,:,None] * illuminant_xyz[None,None,:]
    illuminant_xy = colour.XYZ_to_xy(illuminant_xyz)
    rgb = colour.XYZ_to_RGB(xyz,
                            colourspace=color_space, 
                            apply_cctf_encoding=False,
                            illuminant=illuminant_xy)
    return rgb

def _apply_blur_and_unsharp(self, data):
    unsharp_mask = self.scanner.unsharp_mask
    if config.USE_OPENCL_BLUR and unsharp_mask[0] > 0 and unsharp_mask[1] > 0:
        data = apply_combined_gaussian_unsharp_gpu(data, self.scanner.lens_blur, unsharp_mask)
    elif config.USE_OPENCL_BLUR:
        data = apply_gaussian_blur_gpu(data, self.scanner.lens_blur)
    else:
        data = apply_gaussian_blur(data, self.scanner.lens_blur)
        if unsharp_mask[0] > 0 and unsharp_mask[1] > 0:
            data = apply_unsharp_mask(data, sigma=unsharp_mask[0], amount=unsharp_mask[1])
    return data

def _apply_cctf_encoding_and_clip(self, rgb, color_space, apply_cctf_encoding):
    if apply_cctf_encoding:
        # rgb = colour.cctf_encoding(rgb, function=color_space)
        rgb = colour.RGB_to_RGB(rgb, color_space, color_space,
                apply_cctf_decoding=False,
                apply_cctf_encoding=True)
    rgb = np.clip(rgb, a_min=0, a_max=1)
    return rgb