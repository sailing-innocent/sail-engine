import pytest 
import taichi as ti 
import math

@pytest.mark.app 
def test_bimask():
    ti.init(arch=ti.cpu)

    n = 256
    x = ti.field(ti.f32)
    ti.root.bitmasked(ti.ij, (n, n)).place(x)

    @ti.kernel
    def activate():
        # All elements in bitmasked is initially deactivated
        # Let's activate elements in the rectangle now!
        for i, j in ti.ndrange((100, 125), (100, 125)):
            x[i, j] = 233  # assign any value to activate the element at (i, j)


    @ti.kernel
    def paint_active_pixels(color: ti.f32):
        # struct-for syntax: loop over active pixels, inactive pixels are skipped
        for i, j in x:
            x[i, j] = color


    ti.root.deactivate_all()
    activate()

    gui = ti.GUI('bitmasked', (n, n))
    for frame in range(10000):
        color = math.sin(frame * 0.05) * 0.5 + 0.5
        paint_active_pixels(color)
        #paint_all_pixels(color)  # try this and compare the difference!
        gui.set_image(x)
        gui.show()

    