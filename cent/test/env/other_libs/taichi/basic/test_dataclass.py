import pytest 
import taichi as ti 
import taichi.math as tm 

@pytest.mark.app
def test_rect():
    ti.init(arch=ti.cuda)

    n = 320
    pixels = ti.field(dtype=float, shape=(n, n))

    @ti.dataclass
    class Rect2D:
        pos: tm.vec2
        size: tm.vec2 

    @ti.func 
    def in_rect(p, rect: ti.types.struct()):
       return p[0] > rect.pos[0] and p[1] > rect.pos[1] and p[0] < rect.pos[0] + rect.size[0] and p[1] < rect.pos[1] + rect.size[1]
    
    @ti.kernel
    def paint(t: float):
        rect = Rect2D(tm.vec2(-0.5, -0.5), tm.vec2(0.5, 0.5))
        for i, j in pixels:
            z = tm.vec2(i/n-0.5, j/n-0.5) * 2
            x1 = 1 if in_rect(z, rect) else 0
            pixels[i, j] = x1
        
    gui = ti.GUI("rect test", res=(n, n))

    while gui.running:
        paint(0.03)
        gui.set_image(pixels)
        gui.show()

    assert True