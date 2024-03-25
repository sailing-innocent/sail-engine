import pytest
import drjit as dr 
import mitsuba as mi 

import matplotlib.pyplot as plt

def plot_list(images, title=None):
    fig, axs = plt.subplots(1, len(images), figsize=(14, 4))
    for i in range(len(images)):
        axs[i].imshow(mi.util.convert_to_bitmap(images[i]))
        axs[i].axis('off')
    if title is not None:
        fig.suptitle(title)


lego_scene_file = "D:/data/scenes/mitsuba/lego/scene.xml"

@pytest.mark.app
def test_nerf():
    mi.set_variant('cuda_ad_rgb')
    from mitsuba import ScalarTransform4f as T 

    render_res = 256
    num_stages = 4
    num_iternations_per_stage = 15
    learning_rate = 0.2
    grid_init_res = 16
    
    # spherical harmonic degree to be use for view-dependent appearnce modeling
    sh_degree = 2

    # Enable ReLU in integrator
    use_relu = True

    sensor_count = 7
    sensors = []

    for i in range(sensor_count):
        angle = 360.0 / sensor_count * i 
        sensors.append(mi.load_dict({
            'type': 'perspective',
            'fov': 45,
            'to_world': T.translate([0.5, 0.5, 0.5]) \
                         .rotate([0, 1, 0], angle) \
                         .look_at(target=[0,0,0], origin=[0,0,1.3], up=[0,1,0]),
            'film': {
                'type': 'hdrfilm',
                'width': render_res, 'height': render_res,
                'filter': {'type': 'box'},
                'pixel_format': 'rgba'
            }
        }))

    # rendering synthetic reference images
    scene_ref = mi.load_file(lego_scene_file)
    ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=64) for i in range(sensor_count)]
    plot_list(ref_images, title="Reference images")

    plt.show()

    # TODO pause 2023-08-07

    assert True
