/**
 * @file sph_executor.cpp
 * @brief The SPH Executor
 * @author sailing-innocent
 * @date 2024-05-02
 */
#include "SailInno/solver/csigsph/sph_executor.h"

namespace sail::inno::csigsph {
SPHExecutor::SPHExecutor(SPHSolver& solver) noexcept : m_solver(solver) {
}
const SPHSolver& SPHExecutor::solver() const noexcept {
	return m_solver;
}
SPHSolver& SPHExecutor::solver() noexcept {
	return m_solver;
}
}// namespace sail::inno::csigsph
