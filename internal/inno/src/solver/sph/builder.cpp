#include "SailInno/solver/sph/builder.h"
#include "SailInno/solver/sph/solver.h"
#include "SailInno/solver/sph/fluid_particles.h"

namespace sail::inno::sph {

SPHFluidData SPHFluidBuilder::grid(const luisa::float3& bottom_left,
								   const luisa::float3& grid_size,
								   const float dx2) noexcept {
	SPHFluidData data;
	data.h_pos.clear();
	int n_x = static_cast<int>(grid_size.x / dx2);
	int n_y = static_cast<int>(grid_size.y / dx2);
	int n_z = static_cast<int>(grid_size.z / dx2);

	for (auto ix = 0; ix < n_x; ++ix) {
		for (auto iy = 0; iy < n_y; ++iy) {
			for (auto iz = 0; iz < n_z; ++iz) {
				data.h_pos.push_back(bottom_left + luisa::float3{ix * dx2, iy * dx2, iz * dx2});
			}
		}
	}
	return data;
}

void SPHFluidBuilder::push_particles(const SPHFluidData& data) noexcept {
	m_solver.particles().push_particles(data.h_pos);
}

void SPHFluidBuilder::place_particles(const SPHFluidData& data) noexcept {
	m_solver.particles().place_particles(data.h_pos);
}

}// namespace sail::inno::sph