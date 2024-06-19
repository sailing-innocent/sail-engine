# Simple Root Solver

class RootSolver:
    def __init__(self):
        self.eps = 1e-6
        self.max_iter = 100
        self.create_solver = {
            "newton": self._newton
        }   
        self.set_solver()

    def set_solver(self, name="newton"):
        self.solver = self.create_solver[name]

    def solve(self, f, x0, f_prime):
        return self.solver(f, x0, f_prime)
    

    # Algorithm Implementation
    
    def _newton(self, f, x0, f_prime):
        x = x0
        for i in range(self.max_iter):
            x = x - f(x) / f_prime(x)
            if abs(f(x)) < self.eps:
                return x
        return None
    