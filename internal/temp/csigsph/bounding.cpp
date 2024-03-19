
#include "bounding.h"
#include "sph.h"
#include "sph_executor.h"
#include "fluid_particles.h"

// CORE IMPLEMENTATION
namespace inno::csigsph {
void Bounding::compile() noexcept {
    using namespace luisa;
    using namespace luisa::compute;
    const size_t n_blocks = solver().config().n_blocks;
    const auto dim = 3;
    auto &particles = solver().particles();

    lazy_compile(solver().device(), bounding_cube, [&](Int count, Float delta_time, Float rate) {
        set_block_size(n_blocks);
        grid_stride_loop(count, [&](Int p) noexcept {
            Float3 x = particles.m_pos->read(p);
            Float3 v = particles.m_vel->read(p);

            // Bounding
            auto bmax = make_float3(solver().config().world_size);
            auto bmin = make_float3(0.f);
            Float dist_epsilon = def(1e-7f);
            Float speed_epsilon = 1.f - rate;

            for (int i = 0; i < dim; ++i) {
                auto eps = dist_epsilon;
                $if(x[i] < bmin[i]) {
                    auto old_x = x[i];
                    x[i] = bmin[i] + eps;
                    v[i] *= -speed_epsilon;
                }
                $elif(x[i] > bmax[i]) {
                    auto old_x = x[i];
                    x[i] = bmax[i] - eps;
                    v[i] *= -speed_epsilon;
                };
            }

            particles.m_pos->write(p, x);
            particles.m_vel->write(p, v);
        });
    });

    lazy_compile(solver().device(), bounding_sphere, [&](Int count, Float delta_time, Float rate) {
        set_block_size(n_blocks);
        grid_stride_loop(count, [&](Int p) noexcept {
            Float3 x = particles.m_pos->read(p);
            Float3 v = particles.m_vel->read(p);

            //Sphere Bounding
            auto bmax = make_float3(solver().config().world_size);
            auto bmin = make_float3(0.f);
            Float dist_epsilon = def(1e-7f);
            Float speed_epsilon = 1.f - rate;

            auto sphere_center = bmax / 2.f;
            auto sphere_radius = bmax.x / 1.414f;

            $if(dot(x - sphere_center, x - sphere_center) > sphere_radius * sphere_radius) {
                auto old_x = x;
                auto normal = normalize(x - sphere_center);

                x = normal * (sphere_radius - dist_epsilon) + sphere_center;
                v = v - 2 * dot(v, normal) * normal;
                v = v * speed_epsilon;
            };

            particles.m_pos->write(p, x);
            particles.m_vel->write(p, v);
        });
    });

    lazy_compile(solver().device(), bounding_waterfall, [&](Int count, Float delta_time, Float rate) {
        set_block_size(n_blocks);
        grid_stride_loop(count, [&](Int p) noexcept {
            Float3 x = particles.m_pos->read(p);
            Float3 v = particles.m_vel->read(p);

            //Sphere Bounding
            auto bmax = make_float3(solver().config().world_size);
            auto bmin = make_float3(0.f);
            Float dist_epsilon = def(1e-7f);
            Float speed_epsilon = 1.f - rate;
            auto eps = dist_epsilon;

            $if(x[2] < bmin[2] & x[0]<bmax[0] & x[0]> bmax[0] * 9.f / 10.f) {
                // special bounding
                x[2] = bmax[2] - eps;
                x[0] = x[0] - eps;

                v[0] = 0.0f;
                v[1] = 0.0f;
                v[2] = 0.0f;
            }
            $else {
                // normal bounding
                for (int i = 0; i < dim; ++i) {
                    auto eps = dist_epsilon;
                    $if(x[i] < bmin[i]) {
                        auto old_x = x[i];
                        x[i] = bmin[i] + eps;
                        v[i] *= -speed_epsilon;
                    }
                    $elif(x[i] > bmax[i]) {
                        auto old_x = x[i];
                        x[i] = bmax[i] - eps;
                        v[i] *= -speed_epsilon;
                    };
                };
                // waterfall box
                $if(x[0] > 4 * bmax[0] / 5.f & x[2] < 4 * bmax[2] / 5.f + bmin[2] & x[2] > 3 * bmax[2] / 5.f + bmin[2]) {
                    x[2] = 4 * bmax[2] / 5.f + bmin[2] + eps;
                    v[2] *= -speed_epsilon;
                };
            };
            particles.m_pos->write(p, x);
            particles.m_vel->write(p, v);
        });
    });
    lazy_compile(solver().device(), bounding_heightmap, [&](Int count, Float delta_time, Float rate) {
        set_block_size(n_blocks);
        grid_stride_loop(count,
                         [&](Int p) noexcept {
                             Float3 x = particles.m_pos->read(p);
                             Float3 v = particles.m_vel->read(p);

                             //Sphere Bounding
                             auto bmax = make_float3(solver().config().world_size);
                             auto bmin = make_float3(0.f);
                             Float dist_epsilon = def(1e-7f);
                             Float speed_epsilon = 1.f - rate;
                             auto eps = dist_epsilon;
                             // TODO inlet and outlet

                             // normal bounding
                             for (int i = 0; i < dim; ++i) {
                                 $if(x[i] < bmin[i]) {
                                     auto old_x = x[i];
                                     x[i] = bmin[i] + eps;
                                     v[i] *= -speed_epsilon;
                                 }
                                 $elif(x[i] > bmax[i]) {
                                     auto old_x = x[i];
                                     x[i] = bmax[i] - eps;
                                     v[i] *= -speed_epsilon;
                                 };
                             };
                             // heightmap box
                             auto dx = x[0] - 0.5f;
                             auto dy = x[2] - 0.5f;
                             // get i, j from x, y and heightmap meta
                             // get height and normal from heightmap

                             auto h = sqrt(0.25f - dx * dx - dy * dy);
                             //   $if(x[0] > 0.9f * bmax[0]) {
                             //       h = 0.5f * bmax[1];
                             //   };
                             $if(x[1] < h) {
                                 x[1] = h + eps;
                                 auto n = normalize(make_float3(dx, h, dy));
                                 v = v - 2 * dot(v, n) * n;
                                 v = v * speed_epsilon;
                             };

                             particles.m_pos->write(p, x);
                             particles.m_vel->write(p, v);
                         });
    });
}

}// namespace inno::csigsph

// API IMPLEMENTATION
namespace inno::csigsph {
Bounding::Bounding(SPHSolver &solver) noexcept : SPHExecutor{solver} {
}

void Bounding::create() noexcept {
}

void Bounding::reset() noexcept {
}

void Bounding::solve(luisa::compute::CommandList &cmdlist) noexcept {
    using namespace luisa;
    using namespace luisa::compute;
    auto &particles = solver().particles();
    auto &param = solver().param();
    auto &config = solver().config();
    int num_particles = particles.size();

    switch (solver().param().bound_kind) {
        case SPHBoundKind::SPHERE:
            cmdlist << (*bounding_sphere)(num_particles, param.delta_time, param.collision_rate).dispatch(num_particles);
            break;
        case SPHBoundKind::WATERFALL:
            cmdlist << (*bounding_waterfall)(num_particles, param.delta_time, param.collision_rate).dispatch(num_particles);
            break;
        case SPHBoundKind::HEIGHTMAP:
            cmdlist << (*bounding_heightmap)(num_particles, param.delta_time, param.collision_rate).dispatch(num_particles);
            break;
        default:
            cmdlist << (*bounding_cube)(num_particles, param.delta_time, param.collision_rate).dispatch(num_particles);
            break;
    }
}

}// namespace inno::csigsph