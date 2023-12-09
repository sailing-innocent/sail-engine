import pytest
from lib.inno.dummy_diff_render import DummyDiffRender
import torch 
import numpy as np 
import matplotlib.pyplot as plt

def show_img(tensor):
    tensor_np = tensor.detach().cpu().numpy()
    plt.imshow(tensor_np)
    plt.show()

@pytest.mark.current
def test_dummy_diff_render_module():
    model = DummyDiffRender()
    width = 4
    height = 4
    source_img = torch.zeros((height, width, 3), dtype=torch.float32).cuda()
    source_img.requires_grad = True

    target_img = torch.zeros((height, width, 3), dtype=torch.float32).cuda()
    target_img.requires_grad = False

    # optimizer
    optimizer = torch.optim.SGD([source_img], lr=0.05)
    # optimizer = torch.optim.Adam([source_img], lr=0.05)
    show_interval = 5
    N = 30

    for i in range(N):
        # print(source_img.detach().cpu().numpy())
        result_img = model(source_img, height, width)
        # print(result_img.detach().cpu().numpy())
        # print(target_img.detach().cpu().numpy())
        loss = torch.sum((target_img - result_img) ** 2)
        loss.backward()
        
        source_img_grad = source_img.grad.clone()
        # print(f"source_img_grad: {source_img_grad}")
        print(f"loss: {loss.item()}")

        with torch.no_grad():
            optimizer.step()
            optimizer.zero_grad()
            if i % show_interval == 0:
                show_img(source_img)

