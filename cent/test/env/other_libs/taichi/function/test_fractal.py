import taichi as ti 
import pytest 

@pytest.mark.app
def test_fractal():
    ti.init(arch=ti.gpu)

    n = 320 
    pixels = ti.field(ti.f32)
    ti.root.dense(ti.ij, (2*n, n)).place(pixels)

    @ti.func
    def complex_sqr(z):
        return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1]*z[0]*2])

    @ti.kernel 
    def paint(t: ti.f32):
        for i, j in pixels:
            c = ti.Vector([-0.8, ti.sin(t)*0.2])
            z = ti.Vector([float(i)/n-1, float(j)/n-0.5])*2
            iterations = 0 
            while z.norm()<20 and iterations < 50:
                z = complex_sqr(z)+c 
                iterations += 1
            pixels[i,j] = 1 - iterations*0.02

    gui = ti.GUI("Fractal", (n*2,n))

    for i in range(100000):
        paint(i*0.03)
        gui.set_image(pixels)
        gui.show()

