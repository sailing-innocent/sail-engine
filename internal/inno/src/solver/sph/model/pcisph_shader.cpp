#include "SailInno/solver/sph/solver.h"
#include "SailInno/solver/sph/neighbor.h"
#include "SailInno/solver/sph/model/pcisph.h"
#include "SailInno/solver/sph/neighbor_search_loop.h"

namespace sail::inno::sph {

void PCISPH::compile(Device& device) noexcept {
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

	lazy_compile(device, predictPosAndVel,
				 [&](Int count, Float delta_time) {
		set_block_size(n_blocks);
		grid_stride_loop(count,
						 [&](Int p) noexcept {
			Float3 x = particles.m_pos->read(p);
			Float3 v = particles.m_vel->read(p);
			Float3 dv_vis = m_delta_vel_vis->read(p);
			Float3 dv_pres = m_delta_vel_pres->read(p);

			// predict next time's (x, v)
			v = v + (dv_vis + dv_pres) * delta_time;
			x = x + v * delta_time;

			m_predicted_pos->write(p, x);
		});
	});

	lazy_compile(device, neighborSearch_TmpRho,
				 [&](Float mass, Float kpci, Float h_fac, Float rho_0, Int n_grids, Float cell_size) {
		set_block_size(n_blocks);
		task_search(
			neighbor,
			particles,
			n_grids, n_threads, n_cta, cell_size,
			[&](UInt& p, Float3& pos, Float3& vel, Float& w) {
			pos = m_predicted_pos->read(p);
		},
			[&](SMEM_float3_ptr& pos_ptr, SMEM_float3_ptr& vel_ptr, SMEM_float_ptr& w_ptr, SMEM_int_ptr& cell_offset, SMEM_int_ptr& cell_count) {
			pos_ptr = luisa::make_shared<SMEM_float3>(n_blocks);
			cell_offset = luisa::make_shared<SMEM_int>(n_cta9);
			cell_count = luisa::make_shared<SMEM_int>(n_cta9);
		},
			[&](UInt& idx, UInt& read_idx,
				SMEM_float3_ptr& pos_ptr,
				SMEM_float3_ptr& vel_ptr, SMEM_float_ptr& w_ptr) {
			(*pos_ptr)[idx] = m_predicted_pos->read(read_idx);
		},
			[&](Float3& pos_a, Float3& pos_b,
				Float3& vel_a, Float3& vel_b,
				Float& w_a, Float& w_b,
				Float3& res) {
			Float h_fac2 = h_fac * h_fac;
			$if(is_near_pos(pos_a, pos_b, h_fac2)) {
				Float3 x_ab = pos_a - pos_b;
				res.x += mass * (*this->smoothKernel)(x_ab, h_fac);
				// res.y += 1;
			};
		},
			[&](UInt& p, Float3& res) {
			Float rho = res.x;
			Float k_pci = def(kpci);
			Float corrected_pres = m_corrected_pres->read(p);
			Float error_rho = rho - rho_0;
			error_rho = max(error_rho, 0.f);
			corrected_pres += k_pci * error_rho;

			m_rho->write(p, rho);
			m_corrected_pres->write(p, corrected_pres);

			// \frac{P}{\rho^2}
			Float factor = def(0.f);
			$if(rho > 0.f) { factor = corrected_pres / pow(rho, 2.f); };
			m_pres_factor->write(p, factor);
		});
	});

	lazy_compile(device, neighborSearch_CorPres,
				 [&](Float mass, Float h_fac, Int n_grids, Float cell_size) {
		set_block_size(n_blocks);
		task_search(
			neighbor,
			particles,
			n_grids, n_threads, n_cta, cell_size,
			[&](UInt& p, Float3& pos, Float3& vel, Float& w) {
			pos = m_predicted_pos->read(p);
			// pos = particles.m_pos->read(p);
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
			(*pos_ptr)[idx] = m_predicted_pos->read(read_idx);
			// (*pos_ptr)[idx] = particles.m_pos->read(read_idx);
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
}
}// namespace sail::inno::sph