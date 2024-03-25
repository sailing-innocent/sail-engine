import pytest
import mitsuba as mi 


@pytest.mark.func
def test_cbox_scalar():
    mi.set_variant('scalar_rgb') # scalar_rg, scalar_spectral, cuda_ad_rgb, llvm_ad_rgb
    scene = mi.load_dict(mi.cornell_box())
    img = mi.render(scene)
    mi.Bitmap(img).write('cbox.exr')
    assert True


@pytest.mark.func 
def test_cbox_cuda():
    mi.set_variant('cuda_ad_rgb')
    scene = mi.load_dict(mi.cornell_box())
    img = mi.render(scene)
    mi.Bitmap(img).write('cbox_cuda.exr')
    assert True

@pytest.mark.func
def test_with_matplotlib():
    cbox_scene_file = "D:/data/scenes/mitsuba/cbox.xml"
    mi.set_variant('cuda_ad_rgb')
    scene = mi.load_file(cbox_scene_file)
    img = mi.render(scene, spp=256)

    import matplotlib.pyplot as plt
    plt.axis('off')
    plt.imshow(img ** (1.0 / 2.2))

    plt.show()

    mi.util.write_bitmap("my_first_render.png", image)
    mi.util.write_bitmap("my_first_render.exr", image)