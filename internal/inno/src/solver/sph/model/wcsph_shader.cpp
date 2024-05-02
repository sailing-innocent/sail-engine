#include "SailInno/solver/sph/solver.h"
#include "SailInno/solver/sph/neighbor.h"
#include "SailInno/solver/sph/model/wcsph.h"
#include "SailInno/solver/sph/neighbor_search_loop.h"

namespace sail::inno::sph {
void WCSPH::compile(Device& device) noexcept {
	BaseSPH::compile(device);
	using namespace luisa;
	using namespace luisa::compute;
	const auto dim = 3;
	const size_t n_blocks = solver().config().n_blocks;
	const auto n_threads = solver().config().n_threads;
	const int n_cta = n_blocks / n_threads;
	const size_t n_cta9 = n_cta * 9;

	auto& neighbor = solver().neighbor();
	auto& particles = solver().particles();

	auto is_near_pos = [&](auto x_a, auto x_b, auto h_fac2) noexcept {
		auto x_ab = x_a - x_b;
		Float r2 = length_squared(x_ab);
		Bool res = def(false);
		$if(r2 <= h_fac2) {
			res = def(true);
		};
		return res;
	};

	lazy_compile(device, updatePres,
				 [&](Int count, Float h_fac, Float alpha, Float stiffB, Float gamma, Float rho_0) {
		set_block_size(n_blocks);
		grid_stride_loop(count,
						 [&](Int p) noexcept {
			auto rho = m_rho->read(p);
			// rho = max(rho, rho_0);
			// tait_function
			auto pres = stiffB * (pow(1.f * rho / rho_0, gamma) - 1.0f);
			pres = max(pres, 0.f);// free-surface
			m_pres->write(p, pres);
			// \frac{P}{\rho^2}
			auto factor = pres / pow(rho, 2.f);
			m_pres_factor->write(p, factor);
		});
	});

	lazy_compile(device, neighborSearch_Pres,
				 [&](Float mass, Float h_fac, Int n_grids, Float cell_size) {
		set_block_size(n_blocks);
		task_search(
			neighbor,
			particles,
			n_grids, n_threads, n_cta, cell_size,
			[&](UInt& p, Float3& pos, Float3& vel, Float& w) {
			pos = particles.m_pos->read(p);
			w = m_pres_factor->read(p);
		},
			[&](SMEM_float3_ptr& pos_ptr, SMEM_float3_ptr& vel_ptr, SMEM_float_ptr& w_ptr, SMEM_int_ptr& cell_offset, SMEM_int_ptr& cell_count) {
			pos_ptr = luisa::make_shared<SMEM_float3>(n_blocks);
			w_ptr = luisa::make_shared<SMEM_float>(n_blocks);
			cell_offset = luisa::make_shared<SMEM_int>(n_cta9);
			cell_count = luisa::make_shared<SMEM_int>(n_cta9);
		},
			[&](UInt& idx, UInt& read_idx,
				SMEM_float3_ptr& pos_ptr,
				SMEM_float3_ptr& vel_ptr, SMEM_float_ptr& w_ptr) {
			(*pos_ptr)[idx] = particles.m_pos->read(read_idx);
			(*w_ptr)[idx] = m_pres_factor->read(read_idx);
		},
			[&](Float3& pos_a, Float3& pos_b,
				Float3& vel_a, Float3& vel_b,
				Float& w_a, Float& w_b,
				Float3& res) {
			Float h_fac2 = h_fac * h_fac;
			$if(is_near_pos(pos_a, pos_b, h_fac2)) {
				Float3 x_ab = pos_a - pos_b;
				// Pressure: WCSPH equation (6)
				Float k = -mass * (w_a + w_b);// w = \frac{P}{\rho^2}
				res += k * (*this->smoothGrad)(x_ab, h_fac);
			};
		},
			[&](UInt& p, Float3& res) {
			m_delta_vel_pres->write(p, res);
		});
	});

	lazy_compile(device, forceSearch_Rho,
				 [&](Int count, Float mass, Float h_fac) {
		set_block_size(n_blocks);
		grid_stride_loop(count,
						 [&](Int p) noexcept {
			Float3 pos_a = particles.m_pos->read(p);

			// Float res = mass * (*this->smoothKernel)(pos_a - pos_a, h_fac);
			Float res = 0.f;

			Int cnt = 0;
			$for(j, 0, count) {
				Float3 pos_b = particles.m_pos->read(j);
				Float3 x_ab = pos_a - pos_b;
				$if(is_near_pos(pos_a, pos_b, h_fac * h_fac)) {
					res += mass * (*this->smoothKernel)(x_ab, h_fac);
					cnt += 1;
				};
			};
			m_rho->write(p, res);
			m_delta_vel_vis->write(p, make_float3(0.f));
			m_delta_vel_pres->write(p, make_float3(0.f));
		});
	});

	lazy_compile(device, forceSearch_Force,
				 [&](Int count, Float mass, Float h_fac, Float alpha, Float gamma, Float rho_0, Float3 gravity) {
		set_block_size(n_blocks);
		grid_stride_loop(count,
						 [&](Int p) noexcept {
			Float3 pos_a = particles.m_pos->read(p);
			Float3 vel_a = particles.m_vel->read(p);
			Float pres_a = m_pres_factor->read(p);
			Float rho_a = m_rho->read(p);
			Float h_fac2 = h_fac * h_fac;

			// self
			Float3 res_pres = make_float3(0.f);
			Float3 res_vis = make_float3(0.f);

			$for(j, 0, count) {
				Float3 pos_b = particles.m_pos->read(j);
				$if(is_near_pos(pos_a, pos_b, h_fac2)) {
					Float3 vel_b = particles.m_vel->read(j);
					Float pres_b = m_pres_factor->read(j);
					Float rho_b = m_rho->read(j);
					Float3 x_ab = pos_a - pos_b;
					Float k = -mass * (pres_a + pres_b);// w = \frac{P}{\rho^2}
					// Pressure
					res_pres += k * (*this->smoothGrad)(x_ab, h_fac);

					// Viscosity
					Float3 v_ab = vel_a - vel_b;
					Float v_dot_x = dot(v_ab, x_ab);
					$if(v_dot_x < 0.f & rho_b > 0.f) {
						// Refer to SPlisHSPlasH
						Float mu = 2.f * (dim + 2.f) * alpha;
						Float PI_ab = -mu * (v_dot_x / (length_squared(x_ab) + 0.01f * h_fac2));
						res_vis += -mass / rho_b * PI_ab * (*this->smoothGrad)(x_ab, h_fac);
					};
				};
			};
			// Pressure
			m_delta_vel_pres->write(p, res_pres);
			// Viscosity
			Float3 p_gravity = gravity;// gravity
			m_delta_vel_vis->write(p, res_vis + p_gravity);
			// m_delta_vel_vis->write(p, p_gravity);
		});
	});
}

}// namespace sail::inno::sph