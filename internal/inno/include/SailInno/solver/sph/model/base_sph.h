#pragma once
/**
 * @file source/package/solver/fluid/sph/model/base_sph.h
 * @author sailing-innocent
 * @date 2023-02-23
 * @brief SPH Fluid Model
 */

#include "../sph_executor.h"
namespace sail::inno::sph {
class SPHSolver;

class SAIL_INNO_API BaseSPH : public SPHExecutor {
protected:
	friend class SPHSolver;
	template<size_t I, typename... Ts>
	using Shader = luisa::compute::Shader<I, Ts...>;
	using Int = luisa::compute::Int;
	using Float = luisa::compute::Float;

public:
	BaseSPH(SPHSolver& solver) noexcept;

protected:
	virtual void create(Device& device) noexcept;
	virtual void compile(Device& device) noexcept;
	virtual void iteration(CommandList& cmdlist) noexcept;

protected:
	U<Shader<1, int, float, float>> ms_update_state;
};

}// namespace sail::inno::sph