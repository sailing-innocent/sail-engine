#include "SailInno/solver/sph/model/base.h"
#include "SailInno/solver/sph/solver.h"
// API IMPLEMENTATION
namespace sail::inno::sph {

BaseSPH::BaseSPH(SPHSolver& solver) noexcept : SPHExecutor{solver} {
}

void BaseSPH::reset() noexcept {
	init_mass();
	init_kpci();
	m_size = solver().particles().size();
	// LUISA_INFO("BaseSPH Size {}", m_size);
}

void BaseSPH::create(Device& device) noexcept {
	using namespace luisa;
	using namespace luisa::compute;
	init_cubic();

	init_mass();
	init_kpci();

	m_size = solver().particles().size();
	// LUISA_INFO("BaseSPH Size {}", m_size);
	m_capacity = solver().config().n_capacity;
	allocate(device, m_capacity);
}

void BaseSPH::init_mass() noexcept {
	using namespace luisa;
	using namespace luisa::compute;
	// init mass
	float prefer_rho = 0.f;
	float h = solver().param().h_fac;
	float sigma = (8.f / (PI)) / (h * h * h);// dim = 3
	float dx2 = solver().param().dx * 2;
	float h2 = h * h;
	int num_particles = int(h / dx2);
	// refer to WCSPH(4)
	for (auto ix = -num_particles; ix <= num_particles; ix++) {
		for (auto iy = -num_particles; iy <= num_particles; iy++) {
			for (auto iz = -num_particles; iz <= num_particles; iz++) {
				float3 x_b = make_float3(ix * dx2, iy * dx2, iz * dx2);
				float3 x_ab = -x_b;
				float r2 = x_ab.x * x_ab.x + x_ab.y * x_ab.y + x_ab.z * x_ab.z;
				if (r2 <= h2) {
					//cubicKernel
					float r_len = sqrt(r2);
					float q = r_len / h;

					if (q <= 0.5f) {
						float q2 = q * q;
						prefer_rho += 1.f * sigma * (6.f * (q2 * q - q2) + 1.f);
					} else if (q < 1.0f) {
						prefer_rho += 1.f * sigma * 2.f * (pow(1.f - q, 3.f));
					}
				}
			}
		}
	}

	m_mass = solver().param().rho_0 / prefer_rho;
	// LUISA_INFO("mass: {}", m_mass);
}

void BaseSPH::init_kpci() noexcept {
	using namespace luisa;
	using namespace luisa::compute;
	float delta_time = solver().param().delta_time;
	float rho_0 = solver().param().rho_0;
	float dx2 = solver().param().dx * 2;
	float h = solver().param().h_fac;
	float h2 = h * h;
	int num_particles = int(h / dx2);

	float beta = 2.0f * (delta_time * delta_time) * (m_mass * m_mass) / (rho_0 * rho_0);
	float sigma = (8.f / (PI)) / (h * h * h);// dim = 3

	float3 x_a = make_float3(0.f);
	float3 sum_grad = make_float3(0.f);
	float sum_dot = 0.f;
	int cnt = 0;
	for (auto ix = -num_particles; ix <= num_particles; ix++) {
		for (auto iy = -num_particles; iy <= num_particles; iy++) {
			for (auto iz = -num_particles; iz <= num_particles; iz++) {
				float3 x_b = make_float3(ix * dx2, iy * dx2, iz * dx2);
				float3 x_ab = x_a - x_b;
				float r2 = x_ab.x * x_ab.x + x_ab.y * x_ab.y + x_ab.z * x_ab.z;
				if (r2 <= h2) {
					float r_len = sqrt(r2);

					if (r_len > 1e-6) {
						float3 r_dir = x_ab / r_len;
						float q = r_len / h;
						float3 grad = make_float3(0.f);
						// cubicGrad
						if (q < 0.5f) {
							float q2 = q * q;
							grad = (sigma / h) * (6.f * (3.f * q2 - 2.f * q)) * r_dir;
						} else if (q < 1.0f) {
							grad = -(sigma / h) * 6.f * (pow(1.f - q, 2.f)) * r_dir;
						}
						sum_grad += grad;
						sum_dot += dot(grad, grad);
						cnt++;
					}
				}
			}
		}
	}

	m_kpci = -1.f / (beta * (-dot(sum_grad, sum_grad) - sum_dot));
	// LUISA_INFO("kpci: {}", m_kpci);
	// LUISA_INFO("Prefect Sample: {}", cnt);
}

void BaseSPH::init_cubic() noexcept {
	using namespace luisa;
	using namespace luisa::compute;
	// cubic
	smoothKernel = luisa::make_unique<Callable<float(float3, float)>>(
		[&](Float3 r, Float h) noexcept {
		Float res = def(0.f);
		Float r_len = length(r);
		Float sigma = (8.f / (PI)) / (h * h * h);// dim = 3
		// : (40.f / (7.0f * PI) ) / (h * h); // dim = 2
		Float q = r_len / h;

		$if(q <= 0.5f) {
			Float q2 = q * q;
			res = sigma * (6.f * (q2 * q - q2) + 1.f);
		}
		$elif(q < 1.0f) {
			res = sigma * 2.f * (pow(1.f - q, 3.f));
		};
		return res;
	});

	smoothGrad = luisa::make_unique<Callable<float3(float3, float)>>(
		[&](Float3 r, Float h) noexcept {
		Float r_len = length(r);
		Float3 res = make_float3(0.f);

		$if(r_len > 1e-6f) {
			Float3 r_dir = normalize(r);
			Float sigma = (8.f / (PI)) / (h * h * h);// dim = 3
			// : (40.f / (7.0f * PI) ) / (h * h); // dim = 2
			Float q = r_len / h;

			$if(q < 0.5f) {
				Float q2 = q * q;
				res = (sigma / h) * (6.f * (3.f * q2 - 2.f * q)) * r_dir;
			}
			$elif(q < 1.0f) {
				res = -(sigma / h) * 6.f * (pow(1.f - q, 2.f)) * r_dir;
			};
		};
		return res;
	});
}

void BaseSPH::allocate(Device& device, size_t size) noexcept {
	// LUISA_INFO("BaseSPH Allocate Size:{}", size);
	m_rho = device.create_buffer<float>(size);
	m_pres = device.create_buffer<float>(size);
	m_corrected_pres = device.create_buffer<float>(size);
	m_delta_vel_vis = device.create_buffer<luisa::float3>(size);
	m_delta_vel_pres = device.create_buffer<luisa::float3>(size);
	m_pres_factor = device.create_buffer<float>(size);
	// LUISA_INFO("BaseSPH Allocate Done");
}

void BaseSPH::before_iter(CommandList& cmdlist) noexcept {
	auto n_particles = m_size;
	auto half_fac = solver().param().h_fac * 0.55f;
}

void BaseSPH::after_iter(luisa::compute::CommandList& cmdlist) noexcept {
}
void BaseSPH::iteration(luisa::compute::CommandList& cmdlist) noexcept {
}
void BaseSPH::predict(luisa::compute::CommandList& cmdlist) noexcept {
}

}// namespace sail::inno::sph