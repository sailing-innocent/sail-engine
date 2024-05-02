#include "SailInno/solver/sph/solver.h"
#include "SailInno/solver/sph/neighbor.h"
#include "SailInno/solver/sph/model/base.h"
#include "SailInno/solver/sph/neighbor_search_loop.h"

namespace sail::inno::sph {

void BaseSPH::compile(Device& device) noexcept {
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

	lazy_compile(device, neighborSearch_Rho,
				 [&](Float mass, Float h_fac, Float alpha, Float stiffB, Float gamma, Float rho_0, Int n_grids, Float cell_size) {
		set_block_size(n_blocks);
		task_search(
			neighbor,
			particles,
			n_grids, n_threads, n_cta, cell_size,
			[&](UInt& p, Float3& pos, Float3& vel, Float& w) {
			pos = particles.m_pos->read(p);
		},
			[&](SMEM_float3_ptr& pos_ptr, SMEM_float3_ptr& vel_ptr, SMEM_float_ptr& w_ptr, SMEM_int_ptr& cell_offset, SMEM_int_ptr& cell_count) {
			pos_ptr = luisa::make_shared<SMEM_float3>(n_blocks);
			cell_offset = luisa::make_shared<SMEM_int>(n_cta9);
			cell_count = luisa::make_shared<SMEM_int>(n_cta9);
		},
			[&](UInt& idx, UInt& read_idx,
				SMEM_float3_ptr& pos_ptr,
				SMEM_float3_ptr& vel_ptr, SMEM_float_ptr& w_ptr) {
			(*pos_ptr)[idx] = particles.m_pos->read(read_idx);
		},
			[&](Float3& pos_a, Float3& pos_b,
				Float3& vel_a, Float3& vel_b,
				Float& w_a, Float& w_b,
				Float3& res) {
			$if(is_near_pos(pos_a, pos_b, h_fac * h_fac)) {
				Float3 x_ab = pos_a - pos_b;
				res.x += mass * (*this->smoothKernel)(x_ab, h_fac);
				res.y += 1;
			};
		},
			[&](UInt& p, Float3& res) {
			Float rho = res.x;
			m_rho->write(p, rho);
			// clear
			m_corrected_pres->write(p, 0.f);
		});
	});

	lazy_compile(device, neighborSearch_Vis,
				 [&](Float mass, Float h_fac, Float alpha, Float stiffB, Float gamma, Float rho_0, Float3 gravity, Int n_grids, Float cell_size) {
		set_block_size(n_blocks);
		task_search(
			neighbor,
			particles,
			n_grids, n_threads, n_cta, cell_size,
			[&](UInt& p, Float3& pos, Float3& vel, Float& w) {
			pos = particles.m_pos->read(p);
			vel = particles.m_vel->read(p);
			w = m_rho->read(p);
		},
			[&](SMEM_float3_ptr& pos_ptr, SMEM_float3_ptr& vel_ptr, SMEM_float_ptr& w_ptr, SMEM_int_ptr& cell_offset, SMEM_int_ptr& cell_count) {
			pos_ptr = luisa::make_shared<SMEM_float3>(n_blocks);
			vel_ptr = luisa::make_shared<SMEM_float3>(n_blocks);
			w_ptr = luisa::make_shared<SMEM_float>(n_blocks);
			cell_offset = luisa::make_shared<SMEM_int>(n_cta9);
			cell_count = luisa::make_shared<SMEM_int>(n_cta9);
		},
			[&](UInt& idx, UInt& read_idx,
				SMEM_float3_ptr& pos_ptr,
				SMEM_float3_ptr& vel_ptr, SMEM_float_ptr& w_ptr) {
			(*pos_ptr)[idx] = particles.m_pos->read(read_idx);
			(*vel_ptr)[idx] = particles.m_vel->read(read_idx);
			(*w_ptr)[idx] = m_rho->read(read_idx);
		},
			[&](Float3& pos_a, Float3& pos_b,
				Float3& vel_a, Float3& vel_b,
				Float& w_a, Float& w_b,
				Float3& res) {
			Float h_fac2 = h_fac * h_fac;
			$if(is_near_pos(pos_a, pos_b, h_fac2)) {
				Float3 x_ab = pos_a - pos_b;
				Float3 v_ab = vel_a - vel_b;
				Float v_dot_x = dot(v_ab, x_ab);
				Float rho_b = w_b;
				$if(v_dot_x < 0.f & rho_b > 0.f) {
					// Refer to SPlisHSPlasH
					Float mu = 2.f * (dim + 2.f) * alpha;
					Float PI_ab = -mu * (v_dot_x / (length_squared(x_ab) + 0.01f * h_fac2));
					res += -mass / rho_b * PI_ab * (*this->smoothGrad)(x_ab, h_fac);
				};
			};
		},
			[&](UInt& p, Float3& res) {
			Float3 p_gravity = gravity;// gravity
			m_delta_vel_vis->write(p, res + p_gravity);
			// m_delta_vel_vis->write(p, gravity);
			m_delta_vel_pres->write(p, make_float3(0.f));// clear
		});
	});

	lazy_compile(device, updateStates,
				 [&](Int count, Float delta_time, Float rate) {
		set_block_size(n_blocks);
		grid_stride_loop(count,
						 [&](Int p) noexcept {
			Float3 x = particles.m_pos->read(p);
			Float3 v = particles.m_vel->read(p);
			Float3 dv_vis = m_delta_vel_vis->read(p);
			Float3 dv_pres = m_delta_vel_pres->read(p);

			v = v + (dv_vis + dv_pres) * delta_time;
			x = x + v * delta_time;

			particles.m_pos->write(p, x);
			particles.m_vel->write(p, v);
		});
	});
}

}// namespace sail::inno::sph