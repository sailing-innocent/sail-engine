import pytest 
from .dummy_diff_render import DummyDiffRender
import torch 
import matplotlib.pyplot as plt

@pytest.mark.current 
def test_dummy_diff_render():
    dummy_diff_render = DummyDiffRender()
    height = 100
    width = 100

    source_img = torch.zeros((3, height, width), dtype=torch.float32).cuda()
    source_img[0, :, :] = 1 # set initial image to red

    # target image is blue+green = cyan
    target_img = dummy_diff_render.forward(source_img, height, width)
    target_img_np = target_img.cpu().detach().numpy().transpose(1, 2, 0)
    plt.imshow(target_img_np)
    plt.show()

    source_img = torch.zeros((3, height, width), dtype=torch.float32).cuda()
    source_img[1, :, :] = 1
    source_img.requires_grad = True

    N_ITER = 300
    N_LOG = 50

    optim = torch.optim.Adam([source_img], lr=0.01)

    for i in range(N_ITER):
        result_img = dummy_diff_render.forward(source_img, height, width)
        loss = torch.nn.functional.mse_loss(target_img, result_img)
        loss.backward()

        with torch.no_grad():
            optim.step()
            optim.zero_grad()
            if i % N_LOG == 0:
                print(f'Iter {i}, Loss {loss.item()}')
                result_img_np = result_img.cpu().detach().numpy().transpose(1, 2, 0)
                plt.imshow(result_img_np)
                plt.show()
