/**
 * @file package/diff_gs_projector/projector_impl.cpp
 * @author sailing-innocent
 * @date 2024-03-07
 * @brief The Gaussian Splatter Basic Implement
 */

#include "SailInno/gaussian/diff_gs_projector.h"
#include "SailInno/util/math/gaussian.h"

#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

namespace sail::inno::gaussian {

void DiffGaussianProjector::create(Device& device) noexcept {
	mp_buffer_filler = luisa::make_shared<BufferFiller>();
	compile(device);
}

void DiffGaussianProjector::create(Device& device, S<BufferFiller> p_buffer_filler) noexcept {
	mp_buffer_filler = p_buffer_filler;
	compile(device);
}

void DiffGaussianProjector::forward_impl(
	Device& device,
	CommandList& cmdlist,
	int num_gaussians,
	int sh_deg,
	int max_sh_deg,
	float scale_modifier,
	// input // scene
	BufferView<float> xyz_buffer,
	BufferView<float> feature_buffer,// for color
	BufferView<float> scale_buffer,
	BufferView<float> rotq_buffer,
	// input // camera
	Camera& cam,
	// output // screen space
	BufferView<float> means_2d,
	BufferView<float> depth_features,
	BufferView<float> color_features,
	BufferView<float> covs_2d) {
	// LUISA_INFO("DiffGaussianProjector::forward_impl");
	cmdlist << (*m_forward_preprocess_shader)(
				   num_gaussians, sh_deg, max_sh_deg,
				   // input
				   xyz_buffer,
				   feature_buffer,
				   scale_buffer,
				   rotq_buffer,
				   // params
				   scale_modifier,
				   // output
				   means_2d,
				   depth_features,
				   color_features,
				   covs_2d,
				   // camera
				   cam.pos(),
				   cam.view_matrix())
				   .dispatch(num_gaussians);
}

void DiffGaussianProjector::backward_impl(Device& device) {
}

}// namespace sail::inno::gaussian