import pytest 
import sympy 

@pytest.mark.app
def test_exp_diff():
    x = sympy.symbols('x')
    a = sympy.symbols('a')
    f = sympy.exp(-0.5 * (x**2) / a**2)
    dfdx = sympy.diff(f, x)
    sympy.print_latex(dfdx)

    dfda = sympy.diff(f, a)
    sympy.print_latex(dfda)