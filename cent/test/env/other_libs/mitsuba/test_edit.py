import pytest

import drjit as dr
import mitsuba as mi 

import matplotlib.pyplot as plt

@pytest.mark.current 
def test_edit_scene():
    mi.set_variant('cuda_ad_rgb')
    scene = mi.load_file("D:/data/scenes/mitsuba/simple.xml")
    original_img = mi.render(scene, spp=256)
    # plt.axis('off')
    # plt.imshow(original_img ** (1.0 / 2.2))
    # plt.show()

    params = mi.traverse(scene)
    print(params)
    params['light1.intensity.value'] *= [1.5, 0.2, 0.2]
    params['light2.intensity.value'] *= [0.2, 1.5, 0.2]
    params.update()

    modified_img = mi.render(scene, params=params, spp=256)
    fix, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].axis('off')
    axs[0].imshow(original_img ** (1.0 / 2.2))
    axs[1].axis('off')
    axs[1].imshow(modified_img ** (1.0 / 2.2))
    plt.show()

    assert True 