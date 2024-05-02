#pragma once
/**
 * @file solver.h
 * @brief The SPH Solver
 * @author sailing-innocent
 * @date 2024-05-02
 */

#include "SailInno/core/runtime.h"
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/context.h>

#include "SailInno/helper/buffer_filler.h"
#include "SailInno/solver/csigsph/neighbor.h"
#include "SailInno/solver/csigsph/model.h"
#include "SailInno/solver/csigsph/bounding.h"
#include "SailInno/solver/csigsph/fluid_particles.h"

namespace sail::inno::csigsph {
class FluidParticles;
class BaseSPH;
class Neighbor;
class Bounding;

class SPHSolverConfig {
public:
	// Basic
	int max_iter = 1;
	int n_threads = 32;
	size_t n_blocks = 256;
	float world_size = 1.f;
	size_t n_capacity = 250000;// size = (\frac{grid_size}{2* dx})^ 3 eg: (0.5/(0.004 * 2))^3=244140

	// SPH
	int sph_model_kind = 1;		 // 0 : WCSPH, 1 : PCISPH
	int least_iter = 2;			 // PCISPH correction iteration
	float min_cell_size = 0.016f;// min cell size
	int max_neighbor_count = 100;// max neighbor count
};

enum class SPHBoundKind {
	CUBE = 0,
	SPHERE = 1,
	WATERFALL = 2,
	HEIGHTMAP = 3,
};
class SPHParam {
public:
	luisa::float3 gravity = {0.0f, 0.0f, -9.8f};
	float delta_time = 1.0f / 30.0f;
	SPHBoundKind bound_kind = SPHBoundKind::CUBE;

	float dx = 0.005f;	// particle distance at reference density
	float h_fac = 0.02f;// kernel radius
	float alpha = 0.02f;// viscosity
	float stiffB = 50.f;// pressure stiffness
	float gamma = 7.0f;
	float rho_0 = 1000.f;
	float collision_rate = 0.3f;
};

class SAIL_INNO_API SPHSolver final : public LuisaModule {
	template<typename T>
	using Buffer = luisa::compute::Buffer<T>;

public:
	SPHSolver() noexcept;
	~SPHSolver() noexcept;

	void config(const SPHSolverConfig& config) noexcept;
	const SPHSolverConfig& config() const noexcept { return m_config; }
	void param(const SPHParam& param) noexcept;
	const SPHParam& param() const noexcept { return m_param; }

	void create(Device& device) noexcept;
	void compile(Device& device) noexcept;
	void init_upload(Device& device, CommandList& cmdlist) noexcept;
	void reset(Device& device, CommandList& cmdlist) noexcept;
	void step(Device& device, CommandList& cmdlist) noexcept;
	void setup_iteration(Device& device, CommandList& cmdlist) noexcept;
	void iteration(CommandList& cmdlist) noexcept;
	void finish_iteration(CommandList& cmdlist) noexcept;

	// get
	FluidParticles& particles() noexcept { return *m_particles; }
	Neighbor& neighbor() noexcept { return *m_neighbor; }
	BaseSPH& sphmodel() noexcept { return *m_sphmodel; }
	BufferFiller& filler() noexcept { return *m_filler; }
	DeviceParallel& device_parallel() noexcept { return *m_device_parallel; }
	luisa::compute::Stream& stream() { return *m_stream; }

private:
	friend class SPHInternalTest;
	friend class FluidBuilder;
	SPHSolverConfig m_config;
	SPHParam m_param;
	U<Bounding> m_bounding = nullptr;
	U<FluidParticles> m_particles = nullptr;
	U<Neighbor> m_neighbor = nullptr;
	U<BaseSPH> m_sphmodel = nullptr;
	U<BufferFiller> m_filler = nullptr;
	U<DeviceParallel> m_device_parallel = nullptr;
	luisa::compute::Stream* m_stream = nullptr;
};
}// namespace sail::inno::csigsph
