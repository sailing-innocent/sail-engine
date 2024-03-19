import pytest 
from app.solver.root.simple import RootSolver
import numpy as np

@pytest.mark.current 
def test_newton_solver_1():
    f = lambda x: x**2 - 2
    f_prime = lambda x: 2*x
    x0 = 1.0
    solver = RootSolver()
    x = solver.solve(f, x0, f_prime)
    assert np.isclose(x, np.sqrt(2))