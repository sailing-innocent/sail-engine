import pytest 

from lib.ing.dummy_diff_render import DummyDiffRender
import torch 
import numpy as np 
import matplotlib.pyplot as plt 

def show_compare_imgs(imgs):
    N = len(imgs)
    for i in range(N):
        plt.subplot(1, N, i+1)
        plt.imshow(imgs[i])
    plt.show()

@pytest.mark.current 
def test_dummy_diff_render():
    w = 16
    h = 16
    source_img = torch.zeros((3, h, w), dtype=torch.float32).cuda()
    source_img.requires_grad = True 
    target_img = torch.zeros((3, h, w), dtype=torch.float32).cuda()
    target_img = target_img.detach()

    render = DummyDiffRender()
    result_img = render(source_img, h, w)

    source_img_np = source_img.detach().cpu().numpy().transpose(1, 2, 0)
    target_img_np = target_img.detach().cpu().numpy().transpose(1, 2, 0)
    result_img_np = result_img.detach().cpu().numpy().transpose(1, 2, 0)
    
    show_compare_imgs([source_img_np, target_img_np, result_img_np])


    N_ITERS = 30
    N_SHOW = 10
    optimizer = torch.optim.SGD([source_img], lr=0.1)
    for i in range(N_ITERS):
        result_img = render(source_img, h, w)
        loss = ((target_img - result_img) ** 2).sum()
        loss.backward()

        with torch.no_grad():
            optimizer.step()
            optimizer.zero_grad()
            if (i+1) % N_SHOW == 0:
                result_img_np = result_img.detach().cpu().numpy().transpose(1, 2, 0)
                print(f'Iter {i+1}, loss: {loss.item()}')
                source_img_np = source_img.detach().cpu().numpy().transpose(1, 2, 0)
                show_compare_imgs([source_img_np, target_img_np, result_img_np])