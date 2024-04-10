#ifndef CUDA_SAMPLER
#define CUDA_SAMPLER

#include "cuda_runtime.h"
#include <cuda.h>

namespace SAMPLER {
// Sample the background image.
void panorama_sampler_forward(const float *d_pano, const float *d_dirs,
                              float *d_output, const int w, const int h,
                              const int pano_w, const int pano_h);
void panorama_sampler_backward(const float *d_dLdpix, const float *d_dirs,
                               float *d_dLdpano, const int w, const int h,
                               const int pano_w, const int pano_h);

} // namespace SAMPLER

#endif