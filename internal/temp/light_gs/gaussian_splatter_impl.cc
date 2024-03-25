/**
 * @file package/diff_render/gaussian_splatter_impl.cpp
 * @author sailing-innocent
 * @date 2024-01-03
 * @brief The Light Gaussian Splatter Implement
 */

#include "gaussian_splatter.h"
#include <luisa/dsl/sugar.h>
#include <luisa/backends/ext/cuda/lcub/device_scan.h>
#include <luisa/backends/ext/cuda/lcub/device_radix_sort.h>
#include "util/misc/bit_helper.h"
#include "util/graphic/sh.h"

using namespace luisa;
using namespace luisa::compute;

namespace inno::render
{

void LightGaussianSplatter::compile(Device& device) noexcept
{
    compile_callables(device);
    compile_forward_shade_shader(device);
    compile_forward_preprocess_shader(device);
    compile_copy_with_keys_shader(device);
    compile_get_ranges_shader(device);
    compile_forward_render_shader(device);
}

} // namespace inno::render