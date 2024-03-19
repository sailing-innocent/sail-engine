/**
 * @author Oncle-Ha
 * @date 2023-04-06
 */

#include <luisa/core/logging.h>
#include <luisa/dsl/var.h>

#include "fluid_particles.h"
#include "model.h"
#include "sph.h"
#include "neighbor.h"
#include "neighbor_search_loop.h"

#include <winuser.h>

// #define DEBUG

// CORE IMPLEMENTATION
namespace inno::csigsph {
#ifdef DEBUG
static constexpr auto watch_p = 5;
#endif

void BaseSPH::compile() noexcept {
    using namespace luisa;
    using namespace luisa::compute;

    const auto dim = 3;
    const size_t n_blocks = solver().config().n_blocks;
    const auto n_threads = solver().config().n_threads;
    const int n_cta = n_blocks / n_threads;
    const size_t n_cta9 = n_cta * 9;

    auto &neighbor = solver().neighbor();
    auto &particles = solver().particles();

    auto is_near_pos = [&](auto x_a, auto x_b, auto h_fac2) noexcept {
        auto x_ab = x_a - x_b;
        Float r2 = length_squared(x_ab);
        Bool res = def(false);
        $if(r2 <= h_fac2) {
            res = def(true);
        };
        return res;
    };

    lazy_compile(solver().device(), neighborSearch_Rho,
                 [&](Float mass, Float h_fac, Float alpha, Float stiffB, Float gamma, Float rho_0, Int n_grids, Float cell_size) {
                     set_block_size(n_blocks);
                     task_search(
                         neighbor,
                         particles,
                         n_grids, n_threads, n_cta, cell_size,
                         [&](UInt &p, Float3 &pos, Float3 &vel, Float &w) {
                             pos = particles.m_pos->read(p);
                         },
                         [&](SMEM_float3_ptr &pos_ptr, SMEM_float3_ptr &vel_ptr, SMEM_float_ptr &w_ptr, SMEM_int_ptr &cell_offset, SMEM_int_ptr &cell_count) {
                             pos_ptr = luisa::make_shared<SMEM_float3>(n_blocks);
                             cell_offset = luisa::make_shared<SMEM_int>(n_cta9);
                             cell_count = luisa::make_shared<SMEM_int>(n_cta9);
                         },
                         [&](UInt &idx, UInt &read_idx,
                             SMEM_float3_ptr &pos_ptr,
                             SMEM_float3_ptr &vel_ptr, SMEM_float_ptr &w_ptr) {
                             (*pos_ptr)[idx] = particles.m_pos->read(read_idx);
                         },
                         [&](Float3 &pos_a, Float3 &pos_b,
                             Float3 &vel_a, Float3 &vel_b,
                             Float &w_a, Float &w_b,
                             Float3 &res) {
                             $if(is_near_pos(pos_a, pos_b, h_fac * h_fac)) {
                                 Float3 x_ab = pos_a - pos_b;
                                 res.x += mass * (*this->smoothKernel)(x_ab, h_fac);
                                 res.y += 1;
                             };
                         },
                         [&](UInt &p, Float3 &res) {
                             Float rho = res.x;
            // rho = max(rho, rho_0);
#ifdef DEBUG
                             // $if (p == watch_p)
                             // {
                             solver().printer()->info("[std count] {}: {}", p, res.y);
            // rho = rho_0;
// };
#endif
                             m_rho->write(p, rho);
                             // clear
                             m_corrected_pres->write(p, 0.f);
                         });
                 });

    lazy_compile(solver().device(), neighborSearch_Vis,
                 [&](Float mass, Float h_fac, Float alpha, Float stiffB, Float gamma, Float rho_0, Float3 gravity, Int n_grids, Float cell_size) {
                     set_block_size(n_blocks);
                     task_search(
                         neighbor,
                         particles,
                         n_grids, n_threads, n_cta, cell_size,
                         [&](UInt &p, Float3 &pos, Float3 &vel, Float &w) {
                             pos = particles.m_pos->read(p);
                             vel = particles.m_vel->read(p);
                             w = m_rho->read(p);
                         },
                         [&](SMEM_float3_ptr &pos_ptr, SMEM_float3_ptr &vel_ptr, SMEM_float_ptr &w_ptr, SMEM_int_ptr &cell_offset, SMEM_int_ptr &cell_count) {
                             pos_ptr = luisa::make_shared<SMEM_float3>(n_blocks);
                             vel_ptr = luisa::make_shared<SMEM_float3>(n_blocks);
                             w_ptr = luisa::make_shared<SMEM_float>(n_blocks);
                             cell_offset = luisa::make_shared<SMEM_int>(n_cta9);
                             cell_count = luisa::make_shared<SMEM_int>(n_cta9);
                         },
                         [&](UInt &idx, UInt &read_idx,
                             SMEM_float3_ptr &pos_ptr,
                             SMEM_float3_ptr &vel_ptr, SMEM_float_ptr &w_ptr) {
                             (*pos_ptr)[idx] = particles.m_pos->read(read_idx);
                             (*vel_ptr)[idx] = particles.m_vel->read(read_idx);
                             (*w_ptr)[idx] = m_rho->read(read_idx);
                         },
                         [&](Float3 &pos_a, Float3 &pos_b,
                             Float3 &vel_a, Float3 &vel_b,
                             Float &w_a, Float &w_b,
                             Float3 &res) {
                             Float h_fac2 = h_fac * h_fac;
                             $if(is_near_pos(pos_a, pos_b, h_fac2)) {
                                 Float3 x_ab = pos_a - pos_b;
                                 Float3 v_ab = vel_a - vel_b;
                                 Float v_dot_x = dot(v_ab, x_ab);
                                 Float rho_b = w_b;
                                 $if(v_dot_x<0.f & rho_b> 0.f) {
                                     // Refer to SPlisHSPlasH
                                     Float mu = 2.f * (dim + 2.f) * alpha;
                                     Float PI_ab = -mu * (v_dot_x / (length_squared(x_ab) + 0.01f * h_fac2));
                                     res += -mass / rho_b * PI_ab * (*this->smoothGrad)(x_ab, h_fac);
                                 };
                             };
                         },
                         [&](UInt &p, Float3 &res) {
                             Float3 p_gravity = gravity;// gravity
                             m_delta_vel_vis->write(p, res + p_gravity);
                             // m_delta_vel_vis->write(p, gravity);
                             m_delta_vel_pres->write(p, make_float3(0.f));// clear
                         });
                 });

    lazy_compile(solver().device(), updateStates,
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

void WCSPH::compile() noexcept {
    BaseSPH::compile();

    using namespace luisa;
    using namespace luisa::compute;

    const auto dim = 3;
    const size_t n_blocks = solver().config().n_blocks;
    const auto n_threads = solver().config().n_threads;
    const int n_cta = n_blocks / n_threads;
    const size_t n_cta9 = n_cta * 9;

    auto &neighbor = solver().neighbor();
    auto &particles = solver().particles();

    auto is_near_pos = [&](auto x_a, auto x_b, auto h_fac2) noexcept {
        auto x_ab = x_a - x_b;
        Float r2 = length_squared(x_ab);
        Bool res = def(false);
        $if(r2 <= h_fac2) {
            res = def(true);
        };
        return res;
    };

    lazy_compile(solver().device(), updatePres,
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

    lazy_compile(solver().device(), neighborSearch_Pres,
                 [&](Float mass, Float h_fac, Int n_grids, Float cell_size) {
                     set_block_size(n_blocks);
                     task_search(
                         neighbor,
                         particles,
                         n_grids, n_threads, n_cta, cell_size,
                         [&](UInt &p, Float3 &pos, Float3 &vel, Float &w) {
                             pos = particles.m_pos->read(p);
                             w = m_pres_factor->read(p);
                         },
                         [&](SMEM_float3_ptr &pos_ptr, SMEM_float3_ptr &vel_ptr, SMEM_float_ptr &w_ptr, SMEM_int_ptr &cell_offset, SMEM_int_ptr &cell_count) {
                             pos_ptr = luisa::make_shared<SMEM_float3>(n_blocks);
                             w_ptr = luisa::make_shared<SMEM_float>(n_blocks);
                             cell_offset = luisa::make_shared<SMEM_int>(n_cta9);
                             cell_count = luisa::make_shared<SMEM_int>(n_cta9);
                         },
                         [&](UInt &idx, UInt &read_idx,
                             SMEM_float3_ptr &pos_ptr,
                             SMEM_float3_ptr &vel_ptr, SMEM_float_ptr &w_ptr) {
                             (*pos_ptr)[idx] = particles.m_pos->read(read_idx);
                             (*w_ptr)[idx] = m_pres_factor->read(read_idx);
                         },
                         [&](Float3 &pos_a, Float3 &pos_b,
                             Float3 &vel_a, Float3 &vel_b,
                             Float &w_a, Float &w_b,
                             Float3 &res) {
                             Float h_fac2 = h_fac * h_fac;
                             $if(is_near_pos(pos_a, pos_b, h_fac2)) {
                                 Float3 x_ab = pos_a - pos_b;
                                 // Pressure: WCSPH equation (6)
                                 Float k = -mass * (w_a + w_b);// w = \frac{P}{\rho^2}
                                 res += k * (*this->smoothGrad)(x_ab, h_fac);
                             };
                         },
                         [&](UInt &p, Float3 &res) {
                             m_delta_vel_pres->write(p, res);
                         });
                 });

    lazy_compile(solver().device(), forceSearch_Rho,
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

    lazy_compile(solver().device(), forceSearch_Force,
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
                                                  $if(v_dot_x<0.f & rho_b> 0.f) {
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

void PCISPH::compile() noexcept {
    BaseSPH::compile();

    using namespace luisa;
    using namespace luisa::compute;

    const auto dim = 3;
    const size_t n_blocks = solver().config().n_blocks;
    const auto n_threads = solver().config().n_threads;
    const int n_cta = n_blocks / n_threads;
    const size_t n_cta9 = n_cta * 9;

    auto &neighbor = solver().neighbor();
    auto &particles = solver().particles();

    auto is_near_pos = [&](auto x_a, auto x_b, auto h_fac2) noexcept {
        auto x_ab = x_a - x_b;
        Float r2 = length_squared(x_ab);
        Bool res = def(false);
        $if(r2 <= h_fac2) {
            res = def(true);
        };
        return res;
    };

    lazy_compile(solver().device(), predictPosAndVel,
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

    lazy_compile(solver().device(), neighborSearch_TmpRho,
                 [&](Float mass, Float kpci, Float h_fac, Float rho_0, Int n_grids, Float cell_size) {
                     set_block_size(n_blocks);
                     task_search(
                         neighbor,
                         particles,
                         n_grids, n_threads, n_cta, cell_size,
                         [&](UInt &p, Float3 &pos, Float3 &vel, Float &w) {
                             pos = m_predicted_pos->read(p);
                         },
                         [&](SMEM_float3_ptr &pos_ptr, SMEM_float3_ptr &vel_ptr, SMEM_float_ptr &w_ptr, SMEM_int_ptr &cell_offset, SMEM_int_ptr &cell_count) {
                             pos_ptr = luisa::make_shared<SMEM_float3>(n_blocks);
                             cell_offset = luisa::make_shared<SMEM_int>(n_cta9);
                             cell_count = luisa::make_shared<SMEM_int>(n_cta9);
                         },
                         [&](UInt &idx, UInt &read_idx,
                             SMEM_float3_ptr &pos_ptr,
                             SMEM_float3_ptr &vel_ptr, SMEM_float_ptr &w_ptr) {
                             (*pos_ptr)[idx] = m_predicted_pos->read(read_idx);
                         },
                         [&](Float3 &pos_a, Float3 &pos_b,
                             Float3 &vel_a, Float3 &vel_b,
                             Float &w_a, Float &w_b,
                             Float3 &res) {
                             Float h_fac2 = h_fac * h_fac;
                             $if(is_near_pos(pos_a, pos_b, h_fac2)) {
                                 Float3 x_ab = pos_a - pos_b;
                                 res.x += mass * (*this->smoothKernel)(x_ab, h_fac);
                                 // res.y += 1;
                             };
                         },
                         [&](UInt &p, Float3 &res) {
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

    lazy_compile(solver().device(), neighborSearch_CorPres,
                 [&](Float mass, Float h_fac, Int n_grids, Float cell_size) {
                     set_block_size(n_blocks);
                     task_search(
                         neighbor,
                         particles,
                         n_grids, n_threads, n_cta, cell_size,
                         [&](UInt &p, Float3 &pos, Float3 &vel, Float &w) {
                             pos = m_predicted_pos->read(p);
                             // pos = particles.m_pos->read(p);
                             w = m_pres_factor->read(p);
                         },
                         [&](SMEM_float3_ptr &pos_ptr, SMEM_float3_ptr &vel_ptr, SMEM_float_ptr &w_ptr, SMEM_int_ptr &cell_offset, SMEM_int_ptr &cell_count) {
                             pos_ptr = luisa::make_shared<SMEM_float3>(n_blocks);
                             w_ptr = luisa::make_shared<SMEM_float>(n_blocks);
                             cell_offset = luisa::make_shared<SMEM_int>(n_cta9);
                             cell_count = luisa::make_shared<SMEM_int>(n_cta9);
                         },
                         [&](UInt &idx, UInt &read_idx,
                             SMEM_float3_ptr &pos_ptr,
                             SMEM_float3_ptr &vel_ptr, SMEM_float_ptr &w_ptr) {
                             (*pos_ptr)[idx] = m_predicted_pos->read(read_idx);
                             // (*pos_ptr)[idx] = particles.m_pos->read(read_idx);
                             (*w_ptr)[idx] = m_pres_factor->read(read_idx);
                         },
                         [&](Float3 &pos_a, Float3 &pos_b,
                             Float3 &vel_a, Float3 &vel_b,
                             Float &w_a, Float &w_b,
                             Float3 &res) {
                             Float h_fac2 = h_fac * h_fac;
                             $if(is_near_pos(pos_a, pos_b, h_fac2)) {
                                 Float3 x_ab = pos_a - pos_b;
                                 // Pressure: WCSPH equation (6)
                                 Float k = -mass * (w_a + w_b);// w = \frac{P}{\rho^2}
                                 res += k * (*this->smoothGrad)(x_ab, h_fac);
                             };
                         },
                         [&](UInt &p, Float3 &res) {
                             m_delta_vel_pres->write(p, res);
                         });
                 });
}
}// namespace inno::csigsph

// API IMPLEMENTATION
namespace inno::csigsph {
BaseSPH::BaseSPH(SPHSolver &solver) noexcept : SPHExecutor{solver} {
}

void BaseSPH::reset() noexcept {
    init_mass();
    init_kpci();
    m_size = solver().particles().size();
    LUISA_INFO("BaseSPH Size {}", m_size);
}

void BaseSPH::create() noexcept {
    using namespace luisa;
    using namespace luisa::compute;
    init_cubic();

    init_mass();
    init_kpci();

    m_size = solver().particles().size();
    m_capacity = solver().config().n_capacity;
    allocate(solver().device(), m_capacity);
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
    for (auto ix = -num_particles; ix <= num_particles; ix++)
        for (auto iy = -num_particles; iy <= num_particles; iy++)
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
    m_mass = solver().param().rho_0 / prefer_rho;
    LUISA_INFO("mass: {}", m_mass);
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
    for (auto ix = -num_particles; ix <= num_particles; ix++)
        for (auto iy = -num_particles; iy <= num_particles; iy++)
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

    m_kpci = -1.f / (beta * (-dot(sum_grad, sum_grad) - sum_dot));
    LUISA_INFO("kpci: {}", m_kpci);
    LUISA_INFO("Prefect Sample: {}", cnt);
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

void BaseSPH::allocate(luisa::compute::Device &device, size_t size) noexcept {
    m_rho = device.create_buffer<float>(size);
    m_pres = device.create_buffer<float>(size);

    m_corrected_pres = device.create_buffer<float>(size);
    m_delta_vel_vis = device.create_buffer<luisa::float3>(size);
    m_delta_vel_pres = device.create_buffer<luisa::float3>(size);
    m_pres_factor = device.create_buffer<float>(size);
}

void BaseSPH::before_iter(luisa::compute::CommandList &cmdlist) noexcept {
    auto n_particles = m_size;
    auto half_fac = solver().param().h_fac * 0.55f;
    // auto half_fac = solver().param().h_fac;
}

void BaseSPH::after_iter(luisa::compute::CommandList &cmdlist) noexcept {
}
void BaseSPH::iteration(luisa::compute::CommandList &cmdlist) noexcept {
}

void BaseSPH::predict(luisa::compute::CommandList &cmdlist) noexcept {
}

//WCSPH
WCSPH::WCSPH(SPHSolver &solver) noexcept : BaseSPH(solver) {
}

void WCSPH::create() noexcept {
    BaseSPH::create();
}

void WCSPH::allocate(luisa::compute::Device &device, size_t size) noexcept {
}

void WCSPH::iteration(luisa::compute::CommandList &cmdlist) noexcept {
    auto n_particles = m_size;
    auto num_thread = solver().neighbor().m_num_thread_up;
    auto n_grids = solver().neighbor().m_num_grids;
    auto cell_size = solver().neighbor().m_cell_size;

    auto h_fac = solver().param().h_fac;
    auto alpha = solver().param().alpha;
    auto stiffB = solver().param().stiffB;
    auto gamma = solver().param().gamma;
    auto rho_0 = solver().param().rho_0;
    auto gravity = solver().param().gravity;
    auto delta_time = solver().param().delta_time;
    auto rate = solver().param().collision_rate;

    cmdlist << (*neighborSearch_Rho)(m_mass, h_fac, alpha, stiffB, gamma, rho_0, n_grids, cell_size).dispatch(num_thread)
            << (*updatePres)(n_particles, h_fac, alpha, stiffB, gamma, rho_0).dispatch(n_particles)
            << (*neighborSearch_Vis)(m_mass, h_fac, alpha, stiffB, gamma, rho_0, gravity, n_grids, cell_size).dispatch(num_thread)
            << (*neighborSearch_Pres)(m_mass, h_fac, n_grids, cell_size).dispatch(num_thread)
            << (*updateStates)(n_particles, delta_time, rate).dispatch(n_particles);
}

PCISPH::PCISPH(SPHSolver &solver) noexcept : BaseSPH(solver) {
}

void PCISPH::create() noexcept {
    BaseSPH::create();
    int num_particles = m_size;
    allocate(solver().device(), num_particles);
}

void PCISPH::allocate(luisa::compute::Device &device, size_t size) noexcept {
    m_predicted_pos = device.create_buffer<luisa::float3>(size);
    // m_predicted_vel = device.create_buffer<luisa::float3>(size);
}

void PCISPH::iteration(luisa::compute::CommandList &cmdlist) noexcept {
    auto n_particles = m_size;
    auto num_thread = solver().neighbor().m_num_thread_up;
    auto n_grids = solver().neighbor().m_num_grids;
    auto cell_size = solver().neighbor().m_cell_size;
    // LUISA_INFO("PCISPH Cell Size:{}", cell_size);
    auto h_fac = solver().param().h_fac;
    auto rho_0 = solver().param().rho_0;
    auto delta_time = solver().param().delta_time;

    cmdlist << (*predictPosAndVel)(n_particles, delta_time).dispatch(n_particles)
            << (*neighborSearch_TmpRho)(m_mass, m_kpci, h_fac, rho_0, n_grids, cell_size).dispatch(num_thread)
            << (*neighborSearch_CorPres)(m_mass, h_fac, n_grids, cell_size).dispatch(num_thread);
}

void PCISPH::predict(luisa::compute::CommandList &cmdlist) noexcept {
    auto num_thread = solver().neighbor().m_num_thread_up;
    auto n_grids = solver().neighbor().m_num_grids;
    auto cell_size = solver().neighbor().m_cell_size;
    // LUISA_INFO("PCISPH Cell Size:{}", cell_size);
    auto h_fac = solver().param().h_fac;
    auto alpha = solver().param().alpha;
    auto stiffB = solver().param().stiffB;
    auto gamma = solver().param().gamma;
    auto rho_0 = solver().param().rho_0;
    auto gravity = solver().param().gravity;
    cmdlist << (*neighborSearch_Rho)(m_mass, h_fac, alpha, stiffB, gamma, rho_0, n_grids, cell_size).dispatch(num_thread)
            << (*neighborSearch_Vis)(m_mass, h_fac, alpha, stiffB, gamma, rho_0, gravity, n_grids, cell_size).dispatch(num_thread);
}

void PCISPH::after_iter(luisa::compute::CommandList &cmdlist) noexcept {
    auto n_particles = m_size;
    auto delta_time = solver().param().delta_time;
    auto rate = solver().param().collision_rate;

    cmdlist << (*updateStates)(n_particles, delta_time, rate).dispatch(n_particles);
}

}// namespace inno::csigsph
