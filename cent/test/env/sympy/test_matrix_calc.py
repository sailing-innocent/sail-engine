import pytest 
import sympy as sp

@pytest.mark.current 
def test_matrix_calc():
    J00, J02, J10, J12, J22 = sp.symbols('J_{00} J_{02} J_{10} J_{12} J_{22}')
    J = sp.Matrix([[J00, 0, J02], [0, J10, J12], [J02, J12, J22]])
    W00, W01, W02, W10, W11, W12, W20, W21, W22 = sp.symbols('W_{00} W_{01} W_{02} W_{10} W_{11} W_{12} W_{20} W_{21} W_{22}')
    W = sp.Matrix([[W00, W01, W02], [W10, W11, W12], [W20, W21, W22]])
    C00, C01, C02, C11, C12, C22 = sp.symbols('C_{00} C_{01} C_{02} C_{11} C_{12} C_{22}')
    C = sp.Matrix([[C00, C01, C02], [C01, C11, C12], [C02, C12, C22]])

    V = W @ C @ W.T
    # sp.print_latex(V)

    # dVdC = sp.diff(V, C)
    # get [2][2]
    sigma = V[2, 2]
    dVdC = sp.diff(sigma, C)
    sp.print_latex(dVdC)