import pytest 
import drjit as dr 
import mitsuba as mi 
import matplotlib.pyplot as plt
cbox_scene_file = "D:/data/scenes/mitsuba/cbox.xml"
vol_file = "D:/data/scenes/mitsuba/volume.vol"

def mse(x, y):
    return dr.mean(dr.sqr(x - y))

@pytest.mark.app
def test_gradient_based_optimization():
    mi.set_variant('cuda_ad_rgb')
    scene = mi.load_file(cbox_scene_file, res=128, integrator='prb')
    img_ref = mi.render(scene, spp=512)

    # debug img_ref, can see its right wall is red
    # plt.axis('off')
    # plt.imshow(img_ref ** (1.0 / 2.2))
    # plt.show()

    params = mi.traverse(scene)
    key = 'red.reflectance.value'

    # save the original value
    param_ref = mi.Color3f(params[key])

    params[key] = mi.Color3f(0.01, 0.2, 0.9) # change the 'red' to blue
    params.update()

    img_init = mi.render(scene, spp=128) # now the right wall is blue
    # plt.axis('off')
    # plt.imshow(img_init ** (1.0 / 2.2))
    # plt.show()
    mi.util.convert_to_bitmap(img_init)

    opt = mi.ad.Adam(lr=0.05)
    opt[key] = params[key]
    params.update(opt)
    
    iteration_count = 50
    errors = []

    for it in range(iteration_count):
        img = mi.render(scene, params, spp=4)
        loss = mse(img, img_ref)
        dr.backward(loss)
        opt.step()
        opt[key] = dr.clamp(opt[key], 0.0, 1.0)
        params.update(opt)
        # Track the difference between the current color and the true vlaue
        err_ref = dr.sum(dr.sqr(params[key] - param_ref))
        errors.append(err_ref)
        print(f'Iteration {it} - Loss: {loss}, Error: {err_ref[0]:6f}')

    print(f'Optimization Complete, Final error: {errors[-1]}')

    img_final = mi.render(scene, spp=128)
    plt.axis('off')
    plt.imshow(img_final ** (1.0 / 2.2))
    plt.show()

    mi.util.convert_to_bitmap(img_final)

    # plt.plot(errors)
    # plt.xlabel('Iteration'); plt.ylabel('MSE(param)'); plt.title('Parameter error plot');
    # plt.show()

    assert True



@pytest.mark.app
def test_volumetric_inverse_rendering():
    mi.set_variant('cuda_ad_rgb')
    from mitsuba import ScalarTransform4f as T 
    sensor_count = 5
    sensors = []

    for i in range(sensor_count):
        angle = 180.0 / sensor_count * i - 90.0
        sensor_rotation = T.rotate([0, 1, 0], angle)
        sensor_to_world = T.look_at(target=[0,0,0], origin=[0,0,4], up=[0,1,0])
        sensors.append(mi.load_dict({
            'type': 'perspective',
            'fov': 45,
            'to_world': sensor_rotation @ sensor_to_world,
            'film': {
                'type': 'hdrfilm',
                'width': 64, 'height': 64,
                'filter': {'type': 'tent'}
            }
        }))
    
    scene_dict = {
        'type': 'scene',
        'integrator': {'type': 'prbvolpath'},
        'object': {
            'type': 'cube',
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'heterogeneous',
                'sigma_t': {
                    'type': 'gridvolume',
                    'filename': vol_file,
                    'to_world': T.rotate([1, 0, 0], -90).scale(2).translate(-0.5)
                },
                'scale': 40
            }
        },
        'emitter': {'type': 'constant'}
    }
    scene_ref = mi.load_dict(scene_dict)
    ref_spp = 512
    ref_images = [mi.render(scene_ref, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]
    # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
    # for i in range(sensor_count):
    #     axs[i].imshow(mi.util.convert_to_bitmap(ref_images[i]))
    #     axs[i].axis('off')
    # plt.show()

    # optimization

    # settting up target scene
    v_res = 16

    # Modify the scene dictionary
    scene_dict['object'] = {
        'type': 'cube',
        'interior': {
            'type': 'heterogeneous',
            'sigma_t': {
                'type': 'gridvolume',
                'grid': mi.VolumeGrid(dr.full(mi.TensorXf, 0.002, (v_res, v_res, v_res, 1))),
                'to_world': T.translate(-1).scale(2.0)
            },
            'scale': 40.0,
        },
        'bsdf': {'type': 'null'}
    }

    scene = mi.load_dict(scene_dict)

    init_images = [mi.render(scene, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]

    # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
    # for i in range(sensor_count):
    #     axs[i].imshow(mi.util.convert_to_bitmap(init_images[i]))
    #     axs[i].axis('off')

    params = mi.traverse(scene)
    key = 'object.interior_medium.sigma_t.data'
    opt = mi.ad.Adam(lr=0.02)
    opt[key] = params[key]
    params.update(opt)

    iteration_count = 20
    spp = 8

    for it in range(iteration_count):
        total_loss = 0.0
        for sensor_idx in range(sensor_count):
            img = mi.render(scene, params, sensor=sensors[sensor_idx], spp=spp, seed=it)

            # L2 loss function
            loss = dr.mean(dr.sqr(img - ref_images[sensor_idx]))

            # Backpropagate the loss
            dr.backward(loss)

            # Take a gradient step
            opt.step()

            # clamp
            opt[key] = dr.clamp(opt[key], 1e-6, 1.0)

            params.update(opt)
            total_loss += loss

        print(f'Iteration {it} - Loss: {total_loss}')

    intermediate_images = [mi.render(scene, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]

    # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
    # for i in range(sensor_count):
    #     axs[i].imshow(mi.util.convert_to_bitmap(intermediate_images[i]))
    #     axs[i].axis('off')

    # plt.show()
    opt[key] = dr.upsample(opt[key], shape=(64, 64, 64))
    params.update(opt);
    upscale_images = [mi.render(scene, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]

    # fig, axs = plt.subplots(1, sensor_count, figsize=(14, 4))
    # for i in range(sensor_count):
    #     axs[i].imshow(mi.util.convert_to_bitmap(upscale_images[i]))
    #     axs[i].axis('off')

    # continue the optimization
    for it in range(iteration_count):
        total_loss = 0.0
        for sensor_idx in range(sensor_count):
            img = mi.render(scene, params, sensor=sensors[sensor_idx], spp=8*spp, seed=it)
            loss = dr.mean(dr.sqr(img - ref_images[sensor_idx]))
            dr.backward(loss)
            opt.step()
            opt[key] = dr.clamp(opt[key], 1e-6, 1.0)
            params.update(opt)
            total_loss += loss[0]
        print(f"Iteration {it:02d}: error={total_loss:6f}", end='\r')

    final_images = [mi.render(scene, sensor=sensors[i], spp=ref_spp) for i in range(sensor_count)]

    fig, axs = plt.subplots(2, sensor_count, figsize=(14, 6))
    for i in range(sensor_count):
        axs[0][i].imshow(mi.util.convert_to_bitmap(ref_images[i]))
        axs[0][i].axis('off')
        axs[1][i].imshow(mi.util.convert_to_bitmap(final_images[i]))
        axs[1][i].axis('off')

    plt.show()

    assert True