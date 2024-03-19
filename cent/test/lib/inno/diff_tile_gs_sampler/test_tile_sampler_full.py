import pytest 

from lib.inno.diff_gs_tile_sampler import DiffGSTileSampler, DiffGSTileSamplerSettings
import torch 
import matplotlib.pyplot as plt

@pytest.mark.app
def test_tile_sampler_full():
    sampler = DiffGSTileSampler()
    N = 2
    height = 512
    width = 1024
    fov = 60 / 180 * 3.1415926 
    settings = DiffGSTileSamplerSettings(
        width, height, fov) 

    means_2d = torch.zeros((N, 2), dtype=torch.float32).cuda()
    means_2d[0, 0] = -0.5
    means_2d[0, 1] = -0.5
    covs_2d = 0.1 * torch.ones((N, 3), dtype=torch.float32).cuda()
    covs_2d[:, 1] = 0
    depth_features = torch.ones((N, 1), dtype=torch.float32).cuda()
    depth_features[0, 0] = 0
    color_features = torch.ones((N, 3), dtype=torch.float32).cuda()
    color_features[:, 0] = 1.0
    color_features[:, 1] = 0.0
    color_features[:, 2] = 0.0

    opacity_features = 0.5 * torch.ones((N, 1), dtype=torch.float32).cuda()

    target_img = sampler.forward(
        means_2d, covs_2d, depth_features,             
        opacity_features, color_features, 
        settings)
    target_img_np = target_img.detach().cpu().detach().numpy().transpose(1, 2, 0).clip(0, 1)[::-1, :, :]
    target_img.requires_grad = False

    # change all
    means_2d[0, 0] = 0.5
    means_2d[0, 1] = 0.5
    color_features[0, 0] = 0.5
    color_features[0, 1] = 0.5
    color_features[0, 2] = 0.5
    opacity_features = 1.2 * opacity_features
    covs_2d = 0.2 * covs_2d

    opacity_features.requires_grad = True
    color_features.requires_grad = True
    covs_2d.requires_grad = True
    means_2d.requires_grad = True

    # optim = torch.optim.AdamW([color_features], lr= 1e-2)
    # optim = torch.optim.AdamW([color_features, covs_2d], lr= 1e-2)
    optim = torch.optim.AdamW([means_2d, color_features, opacity_features], lr= 1e-2)
    cov_optim = torch.optim.AdamW([covs_2d], lr= 1e-2)

    N_ROUND = 410
    N_SHOW = 100
    N_OPTIM = 10
    COV_DELAY = 100

    for i in range(N_ROUND):
        # make sure covs_2d is positive
        # covs_2d = torch.abs(covs_2d)
        result_img = sampler.forward(
            means_2d, 
            covs_2d, 
            depth_features, 
            opacity_features, 
            color_features, 
            settings)
        # clamp
        # result_img = torch.clamp(result_img, 0, 1)

        loss = torch.nn.functional.mse_loss(target_img, result_img)
        loss.backward()

        with torch.no_grad():
            # print(covs_2d.detach().cpu().numpy())
            print(loss.item())
            if i % N_SHOW == 0:
                result_img_np = result_img.cpu().detach().numpy()
                # CHW -> HWC
                result_img_np = result_img_np.transpose(1, 2, 0)
                # flip y
                result_img_np = result_img_np[::-1, :, :]
                # compare show
                plt.subplot(1, 2, 1)
                plt.imshow(target_img_np)
                plt.subplot(1, 2, 2)
                plt.imshow(result_img_np)
                plt.show()
            
            optim.step()
            optim.zero_grad(set_to_none=True)
            if i > COV_DELAY:
                cov_optim.step()
                cov_optim.zero_grad(set_to_none=True)

    assert True 