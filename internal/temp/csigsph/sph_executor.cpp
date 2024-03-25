/**
 * @author Oncle-Ha
 * @date 2023-04-06
 */
#include "sph_executor.h"

namespace inno::csigsph {
SPHExecutor::SPHExecutor(SPHSolver &solver) noexcept : m_solver(solver) {
}

const SPHSolver &SPHExecutor::solver() const noexcept {
    return m_solver;
}
SPHSolver &SPHExecutor::solver() noexcept {
    return m_solver;
}
}// namespace inno::csigsph
