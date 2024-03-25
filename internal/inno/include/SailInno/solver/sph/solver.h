#pragma once

/**
 * @file solver/fluid/sph/solver.h
 * @author sailing-innocent
 * @date 2023-02-22
 * @brief Fluid Solver
 */

#include "SailInno/core/runtime.h"
#include "luisa/runtime/device.h"

// helpers
#include "SailInno/helper/buffer_filler.h"
#include "SailInno/helper/device_parallel.h"
// components
#include "neighbor.h"

namespace sail::inno::sph {
class SPHFluidParticles;
class BaseSPH;
class Neighbor;
class Bounding;

enum class SPHModelKind {
	DUMMY = 0,
	WCSPH = 1,
	PCISPH = 2
};

enum class SPHBoundaryKind {
	CUBE = 0,
	SPHERE = 1,
	WATERFALL = 2,
	HEIGHTMAP = 3,
};

struct SPHSolverConfig {
	// basic
	int max_iter = 1;
	size_t n_threads = 32;
	size_t n_blocks = 256;
	float world_size = 1.0f;
	size_t n_capacity = 250000;
	// sph algorithm
	SPHModelKind model_kind = SPHModelKind::DUMMY;

	float min_cell_size = 0.016f;
};

struct SPHSolverParam {

	float dx = 0.005f;// particle distance at reference density
	float delta_time = 1.0f / 30.0f;
	float collision_rate = 0.3f;
	// sph param
	float kernel_radius = 0.02f;
	float viscosity = 0.02f;
	float pressure_stiffness = 50.0f;
	float gamma = 7.0f;
	float rho_0 = 1000.0f;
};

// lifecycle
// 1. create `solver = SPHSolver()`
// 2. config `solver.config(config)`

class SAIL_INNO_API SPHSolver {
	using Device = luisa::compute::Device;
	using Stream = luisa::compute::Stream;
	using CommandList = luisa::compute::CommandList;
	template<typename T>
	using Buffer = luisa::compute::Buffer<T>;

public:
	SPHSolver() noexcept;
	~SPHSolver() noexcept;
	// config set & get
	void config(const SPHSolverConfig& config) noexcept;
	const SPHSolverConfig& config() const noexcept { return m_config; }
	// param set & get
	void param(const SPHSolverParam& param) noexcept;
	const SPHSolverParam& param() const noexcept { return m_param; }

	// lifecycle
	void create(Device& device) noexcept;
	void compile(Device& device) noexcept;
	void init_upload(Device& device, CommandList& cmdlist) noexcept;
	void reset(CommandList& cmdlist) noexcept;
	void step(CommandList& cmdlist) noexcept;

	// component getter

	SPHFluidParticles& particles() noexcept { return *mp_particles; }
	BufferFiller& filler() noexcept { return *mp_buffer_filler; }
	DeviceParallel& device_parallel() noexcept { return *mp_device_parallel; }
	Neighbor& neighbor() noexcept { return *mp_neighbor; }

private:
	friend class FluidBuilder;
	SPHSolverConfig m_config;
	SPHSolverParam m_param;
	// component
	U<SPHFluidParticles> mp_particles;
	U<BaseSPH> mp_sph_model;
	U<Neighbor> mp_neighbor;
	U<Bounding> mp_bounding;

	// helpers
	U<BufferFiller> mp_buffer_filler;
	U<DeviceParallel> mp_device_parallel;
};

}// namespace sail::inno::sph