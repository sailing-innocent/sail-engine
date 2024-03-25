import pytest 
import taichi as ti 

@pytest.mark.app 
def test_prime():
    ti.init(arch=ti.cuda)

    @ti.func 
    def is_prime(n: int):
        result = True 
        for k in range(2, int(n**0.5) + 1):
            if n % k == 0:
                result = False 
                break 

        return result

    @ti.kernel 
    def count_primes(n: int) -> int:
        count = 0
        for k in range(2, n):
            if is_prime(k):
                count += 1
            
        return count 

    assert count_primes(1000000) == 78498



@pytest.mark.current 
def test_julia_set():
    ti.init(arch=ti.gpu)

    n = 320
    pixels = ti.field(dtype=float, shape=(n*2, n))

    @ti.func
    def complex_sqr(z): # complex squire of a 2D vector
        return ti.Vector([z[0] * z[0] - z[1] * z[1], 2 * z[0] * z[1]])
    
    @ti.kernel
    def paint(t: float):
        for i, j in pixels:
            c = ti.Vector([-0.8, ti.cos(t) * 0.2])
            z = ti.Vector([i/n-1, j/n-0.5]) * 2
            iterations = 0
            while z.norm() < 20 and iterations < 50:
                z = complex_sqr(z) + c
                iterations += 1
            pixels[i,j] = 1 - iterations * 0.02
    
    gui = ti.GUI("Julia Set", res = (n *2 , n))

    for i in range(100):
        paint(i * 0.03)
        gui.set_image(pixels)
        gui.show()

    assert True