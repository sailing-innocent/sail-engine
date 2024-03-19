# the basic taichi gui
import pytest 
import taichi as ti 
import numpy as np 

@pytest.mark.app
def test_gui_circles():
    ti.init("cuda")
    pos = np.random.random((50, 2))
    # create an array of 50 integer elements whose values are randomly 0, 1, 2
    indices = np.random.randint(0, 2, size=(50,))
    
    gui = ti.GUI("circles", (800, 800))

    while gui.running:
        gui.circles(pos, radius=5, palette=[0x068587, 0xED5538, 0xEEEEF0], palette_indices=indices)
        gui.show()

    assert 0 == 0

@pytest.mark.app
def test_gui_lines():
    ti.init("cuda")
    X = np.random.random((5, 2))
    Y = np.random.random((5, 2))
    gui = ti.GUI("lines", (800, 800))
    while gui.running:
        gui.lines(X, Y, radius=2, color=0x068587)
        gui.show()

@pytest.mark.app
def test_gui_triangles():
    ti.init("cuda")
    X = np.random.random((2, 2))
    Y = np.random.random((2, 2))
    Z = np.random.random((2, 2))
    gui = ti.GUI("triangles", res=(800, 800))

    while gui.running:
        gui.triangles(a=X, b=Y, c=Z, color=0xED5538)
        gui.show()

@pytest.mark.app
def test_ggui():
    ti.init(arch=ti.cuda)

    N = 10

    particles_pos = ti.Vector.field(3, dtype=ti.f32, shape = N)
    points_pos = ti.Vector.field(3, dtype=ti.f32, shape = N)

    @ti.kernel
    def init_points_pos(points : ti.template()):
        for i in range(points.shape[0]):
            points[i] = [i for j in ti.static(range(3))]

    init_points_pos(particles_pos)
    init_points_pos(points_pos)

    window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(5, 2, 2)

    while window.running:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(particles_pos, color = (0.68, 0.26, 0.19), radius = 0.1)
        # Draw 3d-lines in the scene
        scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 5.0)
        canvas.scene(scene)
        window.show()

    assert True