from lib.sailcu.dummy_diff_render import DummyDiffRender
import torch 

import matplotlib.pyplot as plt

def test_dummy_diff_render():
    dummy_diff_render = DummyDiffRender()
    height = 100
    width = 100
    source_img = torch.zeros((3, height, width), dtype=torch.float32).cuda()
    red = torch.ones((1, 100, 100), dtype=torch.float32).cuda()
    source_img[0] = red
    source_img_np = source_img.cpu().detach().numpy().transpose(1, 2, 0)
    plt.imshow(source_img_np)
    plt.show()

    result_img = dummy_diff_render.forward(source_img, height, width)
    # result_img.backward(torch.ones_like(result_img))
    result_img_np = result_img.cpu().detach().numpy().transpose(1, 2, 0)
    
    plt.imshow(result_img_np)
    plt.show()

    assert True